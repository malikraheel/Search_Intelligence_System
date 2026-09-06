"""Planning nodes: query_planner (LLM) → plan_validator (code) → plan_repair (LLM) | fallback_plan (code).
Plus recheck_plan (code) for single-query re-runs and dispatch_retrieval (code) which expands a plan
into concrete tool calls."""

from __future__ import annotations

from typing import Any

from app.domain.models import PlannedCall, RetrievalPlan, RunError, SubQuery
from app.gateway import AllProvidersFailed, Role, StructuredOutputError
from app.graph.deps import Deps
from app.graph.prompts import planner_messages
from app.graph.state import RunState

MAX_PLAN_REPAIRS = 1


def query_planner(state: RunState, deps: Deps) -> dict[str, Any]:
    try:
        resp = deps.gateway.complete(
            role=Role.PLANNER,
            messages=planner_messages(state["profile"], state["question"]),
            run_id=state["run_id"],
            response_model=RetrievalPlan,
        )
    except (AllProvidersFailed, StructuredOutputError) as exc:
        return {
            "plan": None,
            "plan_errors": [f"planner unavailable: {exc}"],
            "plan_repair_attempts": MAX_PLAN_REPAIRS,  # nothing to repair → validator routes to fallback
            "errors": [RunError(node="query_planner", kind=exc.kind, message=str(exc))],
        }
    return {
        "plan": resp.parsed,
        "plan_source": "llm",
        "plan_errors": [],
        "plan_repair_attempts": 0,
        "llm_usage": [resp.usage],
    }


def plan_validator(state: RunState, deps: Deps) -> dict[str, Any]:
    plan = state.get("plan")
    if plan is None:
        return {"plan_errors": state.get("plan_errors") or ["no plan produced"]}

    errors: list[str] = []
    seen: set[str] = set()
    cleaned: list[SubQuery] = []
    for sq in plan.sub_queries:
        text = " ".join(sq.query_text.lower().strip().strip('"').split())
        if len(text) < 3 or len(text) > 120:
            errors.append(f"query_text out of bounds (3-120 chars): '{sq.query_text}'")
            continue
        if text in seen:
            continue  # silent dedupe
        seen.add(text)
        cleaned.append(sq.model_copy(update={"query_text": text}))

    if len(cleaned) < 2:
        errors.append(f"plan needs at least 2 distinct valid sub-queries, got {len(cleaned)}")
    if len(cleaned) > 8:
        errors.append(f"plan has too many sub-queries ({len(cleaned)} > 8)")
    brand = state["profile"].name.lower()
    if cleaned and not any(brand.split()[0] in sq.query_text for sq in cleaned):
        errors.append(f"plan must include at least one brand query containing '{brand}'")

    if errors:
        return {"plan_errors": errors}
    return {"plan": plan.model_copy(update={"sub_queries": cleaned}), "plan_errors": []}


def route_after_plan_validation(state: RunState) -> str:
    if not state.get("plan_errors"):
        return "dispatch_retrieval"
    if state.get("plan") is not None and state.get("plan_repair_attempts", 0) < MAX_PLAN_REPAIRS:
        return "plan_repair"
    return "fallback_plan"


def plan_repair(state: RunState, deps: Deps) -> dict[str, Any]:
    attempts = state.get("plan_repair_attempts", 0) + 1
    try:
        resp = deps.gateway.complete(
            role=Role.PLANNER,
            messages=planner_messages(state["profile"], state["question"], repair_errors=state.get("plan_errors")),
            run_id=state["run_id"],
            response_model=RetrievalPlan,
            use_cache=False,
        )
    except (AllProvidersFailed, StructuredOutputError) as exc:
        return {
            "plan_repair_attempts": attempts,
            "plan_errors": [*(state.get("plan_errors") or []), f"repair failed: {exc}"],
            "errors": [RunError(node="plan_repair", kind=exc.kind, message=str(exc))],
        }
    return {"plan": resp.parsed, "plan_source": "repaired", "plan_repair_attempts": attempts, "llm_usage": [resp.usage]}


def fallback_plan(state: RunState, deps: Deps) -> dict[str, Any]:
    """Deterministic plan built only from the profile — used when the LLM planner is unusable."""
    p = state["profile"]
    industry = p.industry.lower().replace(" software", "").replace(" tools", "").strip() or "software"
    queries = [
        SubQuery(query_text=p.name.lower(), intent="brand", priority=1, wants_ai_check=False),
        SubQuery(query_text=f"best {industry} software", intent="category", priority=1),
        SubQuery(query_text=f"{industry} tools", intent="category", priority=2),
        SubQuery(query_text=f"{p.name.lower()} reviews", intent="informational", priority=3, wants_ai_check=False),
    ]
    for comp in p.competitors[:2]:
        comp_name = comp.split(".")[0].lower()
        queries.append(SubQuery(query_text=f"{p.name.lower()} vs {comp_name}", intent="comparison", priority=2))
    plan = RetrievalPlan(sub_queries=queries[:8], rationale="Deterministic fallback plan derived from the profile.")
    return {
        "plan": plan,
        "plan_source": "fallback",
        "plan_errors": [],
        "degradation": ["planner_fallback: " + "; ".join(state.get("plan_errors") or ["unknown"])],
    }


def recheck_plan(state: RunState, deps: Deps) -> dict[str, Any]:
    text = (state.get("recheck_query") or "").strip().lower()
    plan = RetrievalPlan(
        sub_queries=[SubQuery(query_text=text, intent="category", priority=1, wants_ai_check=True)],
        rationale="Single-query recheck.",
    )
    return {"plan": plan, "plan_source": "recheck", "plan_errors": []}


def dispatch_retrieval(state: RunState, deps: Deps) -> dict[str, Any]:
    """Expand the validated plan into concrete tool calls (1 SERP per query, 1 AI check where wanted,
    ONE batched keyword-volume call for every keyword)."""
    plan = state["plan"]
    assert plan is not None
    calls: list[PlannedCall] = []
    for sq in plan.sub_queries:
        calls.append(PlannedCall(tool="serp_organic", query_text=sq.query_text, intent=sq.intent))
        if sq.wants_ai_check:
            calls.append(PlannedCall(tool="llm_responses", query_text=sq.query_text, intent=sq.intent))
    calls.append(
        PlannedCall(
            tool="keyword_volume", query_text="<all keywords>", keywords=[sq.query_text for sq in plan.sub_queries]
        )
    )
    return {"planned_calls": calls}
