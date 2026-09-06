"""A role-aware fake LLM: produces plausible, schema-valid outputs for every agent role from the
prompt content alone. Tests override individual roles by enqueueing scripted responses."""

from __future__ import annotations

import json
import re
from typing import Any

from app.gateway import Gateway, GatewayConfig, Role, RouteConfig
from app.gateway.providers.fake import FakeProvider, json_response, text_response, tool_call_response
from app.gateway.types import LLMRequest, ProviderResponse
from app.observability.metrics import MetricsRegistry
from app.resilience import BreakerRegistry


def model_for(role: Role) -> str:
    return f"fake/{role.value}"


def _user_text(req: LLMRequest) -> str:
    return next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")


def _json_after(text: str, marker: str) -> Any:
    idx = text.find(marker)
    if idx < 0:
        return None
    return json.loads(text[idx + len(marker) :].strip().split("\n\n")[0])


def smart_llm(req: LLMRequest) -> ProviderResponse:
    role = req.model.split("/")[-1]
    user = _user_text(req)
    if role == "planner":
        profile = _json_after(user, "Profile: ")
        name, industry = profile["brand"].lower(), profile["industry"].lower().replace(" software", "")
        comp = (profile["competitors"] or ["rival.com"])[0].split(".")[0]
        plan = {
            "sub_queries": [
                {"query_text": name, "intent": "brand", "priority": 1, "wants_ai_check": False},
                {
                    "query_text": f"best {industry} software",
                    "intent": "category",
                    "priority": 1,
                    "wants_ai_check": True,
                },
                {"query_text": f"{name} vs {comp}", "intent": "comparison", "priority": 2, "wants_ai_check": True},
                {
                    "query_text": f"how to improve {industry}",
                    "intent": "informational",
                    "priority": 3,
                    "wants_ai_check": True,
                },
            ],
            "rationale": "brand + category + comparison + informational coverage",
        }
        return json_response(plan, tokens=300, model=req.model)
    if role == "retrieval":
        step = _json_after(user, "PLANNED_STEP: ")
        tool = step["expected_tool"]
        if tool == "serp_organic":
            args = {
                "keyword": step["query_text"],
                "location_code": step["location_code"],
                "language_code": step["language_code"],
            }
        elif tool == "llm_responses":
            args = {"user_prompt": f"what is the best option for {step['query_text']}?", "web_search": True}
        else:
            args = {"keywords": step["keywords"], "location_code": step["location_code"]}
        return tool_call_response(tool, args, tokens=80, model=req.model)
    if role == "analyst":
        m = re.search(r"Normalised records \(\d+\):\n(.*)$", user, re.S)
        records = json.loads(m.group(1)) if m else []
        top = sorted(records, key=lambda r: r["opportunity_score"], reverse=True)[:3] or [{"query_text": "x"}]
        analysis = {
            "executive_summary": "Brand has gaps in category and comparison queries; AI citations are inconsistent.",
            "insights": [
                {
                    "query_text": r["query_text"],
                    "headline": f"Opportunity on '{r['query_text']}'",
                    "evidence": f"score={r.get('opportunity_score')}, visible={r.get('domain_visible')}",
                    "relevance_score": 0.8,
                }
                for r in top
            ],
            "recommendations": [
                {
                    "target_query_text": r["query_text"],
                    "content_type": "blog_post",
                    "title": f"Guide: {r['query_text']}",
                    "rationale": "Closes a visibility gap.",
                    "target_keywords": [r["query_text"]],
                    "priority": "high",
                }
                for r in top
            ],
        }
        return json_response(analysis, tokens=500, model=req.model)
    if role == "reporter":
        return text_response(
            "**Summary**\n- The brand ranks for some queries.\n- Improve category coverage.\n- Publish comparisons.",
            tokens=150,
            model=req.model,
        )
    raise RuntimeError(f"smart_llm: unknown role in model {req.model}")


def make_fake_gateway(metrics: MetricsRegistry, provider: FakeProvider | None = None) -> tuple[Gateway, FakeProvider]:
    fake = provider or FakeProvider(default=smart_llm)
    routes = {r: RouteConfig(model=model_for(r), fallbacks=[], max_retries=1) for r in Role}
    cfg = GatewayConfig(routes=routes, retry_base_delay_s=0, cache_enabled=False)
    gw = Gateway(
        cfg, {"fake": fake}, metrics=metrics, breakers=BreakerRegistry(failure_threshold=50, reset_timeout_s=60)
    )
    return gw, fake
