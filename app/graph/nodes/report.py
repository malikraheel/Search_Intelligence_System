"""Report agent: assembles the final structured report (JSON + human-readable summary) and derives status."""

from __future__ import annotations

from typing import Any

from app.domain.models import Report, RunError
from app.gateway import AllProvidersFailed, Role
from app.graph.deps import Deps
from app.graph.prompts import reporter_messages
from app.graph.state import RunState


def derive_status(state: RunState) -> tuple[str, list[str]]:
    reasons = list(dict.fromkeys(state.get("degradation", [])))  # de-dup, keep order
    records = state.get("records", [])
    if not records:
        return "failed", reasons or ["no usable records"]
    for r in state.get("raw_results", []):
        if r.status == "degraded":
            reasons.append(f"degraded:{r.tool}:{r.query_text}")
    if state.get("plan_source") == "fallback":
        reasons.append("plan_source:fallback")
    if state.get("analysis_source") == "fallback":
        reasons.append("analysis_source:fallback")
    if state.get("dropped_records"):
        reasons.append(f"dropped_records:{state['dropped_records']}")
    reasons = list(dict.fromkeys(reasons))
    return ("partial" if reasons else "completed"), reasons


def _usage_totals(state: RunState) -> dict[str, Any]:
    usage = state.get("llm_usage", [])
    return {
        "llm_calls": len(usage),
        "prompt_tokens": sum(u.prompt_tokens for u in usage),
        "completion_tokens": sum(u.completion_tokens for u in usage),
        "total_tokens": sum(u.total_tokens for u in usage),
        "cost_usd": round(sum(u.cost_usd for u in usage), 6),
        "cost_unknown": any(u.cost_unknown for u in usage),
    }


def _deterministic_summary(state: RunState, status: str, reasons: list[str]) -> str:
    p = state["profile"]
    records = state.get("records", [])
    analysis = state.get("analysis")
    visible = [r for r in records if r.domain_visible]
    cited = [r for r in records if r.ai_cited]
    top = sorted(records, key=lambda r: r.opportunity_score, reverse=True)[:3]
    lines = [
        f"**{p.name} visibility report** — status: {status}.",
        f"- Tracked {len(records)} queries; ranking in Google for {len(visible)}, "
        f"cited in AI answers for {len(cited)}.",
    ]
    if top:
        lines.append(
            "- Biggest opportunities: "
            + ", ".join(f"'{r.query_text}' (score {r.opportunity_score:.2f})" for r in top)
            + "."
        )
    if analysis:
        lines.append(f"- {analysis.executive_summary}")
        for rec in analysis.recommendations[:3]:
            lines.append(f"- Recommended: {rec.content_type} — {rec.title} [{rec.priority}]")
    if reasons:
        lines.append(f"- Degradation: {'; '.join(reasons[:5])}")
    return "\n".join(lines)


def _assemble_json(state: RunState, status: str, reasons: list[str]) -> dict[str, Any]:
    p = state["profile"]
    records = state.get("records", [])
    analysis = state.get("analysis")
    return {
        "run_id": state["run_id"],
        "mode": state.get("mode", "full"),
        "status": status,
        "profile": {"uuid": p.profile_uuid, "name": p.name, "domain": p.domain, "industry": p.industry},
        "question": state.get("question"),
        "plan": {
            "source": state.get("plan_source"),
            "sub_queries": [sq.model_dump() for sq in (state["plan"].sub_queries if state.get("plan") else [])],
            "planned_calls": len(state.get("planned_calls", [])),
        },
        "retrieval": {
            "raw_results": len(state.get("raw_results", [])),
            "ok": sum(1 for r in state.get("raw_results", []) if r.status == "ok"),
            "degraded": sum(1 for r in state.get("raw_results", []) if r.status == "degraded"),
            "api_calls": len(state.get("api_calls", [])),
        },
        "records": [
            r.model_dump(mode="json") for r in sorted(records, key=lambda r: r.opportunity_score, reverse=True)
        ],
        "records_extracted": len(records),
        "dropped_records": state.get("dropped_records", 0),
        "analysis": analysis.model_dump(mode="json") if analysis else None,
        "analysis_source": state.get("analysis_source"),
        "degradation": reasons,
        "errors": [e.model_dump() for e in state.get("errors", [])],
        "token_usage": _usage_totals(state),
    }


def report(state: RunState, deps: Deps) -> dict[str, Any]:
    status, reasons = derive_status(state)
    report_json = _assemble_json(state, status, reasons)
    summary = _deterministic_summary(state, status, reasons)
    update: dict[str, Any] = {"status": status}

    # Prose polish via the reporter role; strictly optional — the deterministic summary is the safety net.
    analysis = state.get("analysis")
    if analysis is not None:
        findings = {
            "brand": state["profile"].name,
            "status": status,
            "executive_summary": analysis.executive_summary,
            "insights": [i.model_dump() for i in analysis.insights[:6]],
            "recommendations": [r.model_dump() for r in analysis.recommendations[:5]],
            "degradation": reasons[:5],
        }
        try:
            resp = deps.gateway.complete(
                role=Role.REPORTER, messages=reporter_messages(findings), run_id=state["run_id"]
            )
            if resp.text and len(resp.text.strip()) > 40:
                summary = resp.text.strip()
            update["llm_usage"] = [resp.usage]
            report_json["token_usage"] = _usage_totals(
                {**state, "llm_usage": [*state.get("llm_usage", []), resp.usage]}
            )
        except AllProvidersFailed as exc:
            update["errors"] = [RunError(node="report", kind=exc.kind, message=f"summary LLM unavailable: {exc}")]

    report_json["summary"] = summary
    update["report"] = Report(report_json=report_json, summary=summary)
    return update


def degraded_report(state: RunState, deps: Deps) -> dict[str, Any]:
    status, reasons = derive_status(state)
    status = "failed"
    report_json = _assemble_json(state, status, reasons)
    summary = (
        f"**{state['profile'].name} visibility report** — status: failed. No usable records could be retrieved. "
        f"Reasons: {'; '.join(reasons[:5]) or 'unknown'}."
    )
    report_json["summary"] = summary
    return {"status": status, "report": Report(report_json=report_json, summary=summary)}
