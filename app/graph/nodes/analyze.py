"""Analysis/Synthesis agent (LLM) with a deterministic rule-based fallback."""

from __future__ import annotations

from typing import Any

from app.domain.models import AnalysisResult, Insight, QueryRecord, RecommendationDraft, RunError
from app.gateway import AllProvidersFailed, Role, StructuredOutputError
from app.graph.deps import Deps
from app.graph.prompts import analyst_messages
from app.graph.state import RunState


def _match_query(text: str, records: list[QueryRecord]) -> QueryRecord | None:
    t = text.strip().lower()
    for r in records:
        if r.query_text.lower() == t:
            return r
    for r in records:  # loose containment match
        if t in r.query_text.lower() or r.query_text.lower() in t:
            return r
    return None


def _ground(analysis: AnalysisResult, records: list[QueryRecord]) -> tuple[AnalysisResult, list[str]]:
    """Snap LLM output to real records: fix query_text, copy pre-computed scores, drop unmatched items."""
    issues: list[str] = []
    insights: list[Insight] = []
    for ins in analysis.insights:
        rec = _match_query(ins.query_text, records)
        if rec is None:
            issues.append(f"insight dropped (unknown query): {ins.query_text!r}")
            continue
        insights.append(
            ins.model_copy(update={"query_text": rec.query_text, "opportunity_score": rec.opportunity_score})
        )
    recs: list[RecommendationDraft] = []
    for rd in analysis.recommendations:
        rec = _match_query(rd.target_query_text, records)
        if rec is None:
            issues.append(f"recommendation dropped (unknown query): {rd.target_query_text!r}")
            continue
        recs.append(rd.model_copy(update={"target_query_text": rec.query_text}))
    if not insights:  # keep the schema promise (min 1 insight) via rules
        insights = _rule_insights(records)[:3]
    return analysis.model_copy(update={"insights": insights, "recommendations": recs}), issues


def analyze(state: RunState, deps: Deps) -> dict[str, Any]:
    records = state.get("records", [])
    try:
        resp = deps.gateway.complete(
            role=Role.ANALYST,
            messages=analyst_messages(state["profile"], state["question"], records),
            run_id=state["run_id"],
            response_model=AnalysisResult,
        )
    except (AllProvidersFailed, StructuredOutputError) as exc:
        return {
            "analysis": None,
            "errors": [RunError(node="analyze", kind=exc.kind, message=str(exc))],
            "_trace": {"status": "degraded", "error": str(exc)[:300]},
        }
    grounded, issues = _ground(resp.parsed, records)
    update: dict[str, Any] = {"analysis": grounded, "analysis_source": "llm", "llm_usage": [resp.usage]}
    if issues:
        update["errors"] = [RunError(node="analyze", kind="grounding", message=i) for i in issues]
    return update


def route_after_analyze(state: RunState) -> str:
    return "report" if state.get("analysis") is not None else "analyze_fallback"


def _rule_insights(records: list[QueryRecord]) -> list[Insight]:
    out: list[Insight] = []
    for r in sorted(records, key=lambda x: x.opportunity_score, reverse=True):
        if r.domain_visible is False:
            headline = f"Not ranking for '{r.query_text}'"
        elif r.domain_visible and (r.visibility_position or 1) > 3:
            headline = f"Ranking #{r.visibility_position} for '{r.query_text}' — room to move up"
        elif r.ai_cited is False:
            headline = f"Absent from AI answers for '{r.query_text}'"
        else:
            headline = f"Strong position for '{r.query_text}'"
        evidence = (
            f"volume={r.estimated_search_volume}, difficulty={r.competitive_difficulty}, "
            f"position={r.visibility_position}, ai_cited={r.ai_cited}, "
            f"competitors_in_top10={len(r.competitor_positions)}"
        )
        out.append(
            Insight(
                query_text=r.query_text,
                headline=headline,
                evidence=evidence,
                relevance_score=round(min(1.0, 0.5 + r.opportunity_score / 2), 3),
                opportunity_score=r.opportunity_score,
            )
        )
    return out


def _rule_recommendations(records: list[QueryRecord]) -> list[RecommendationDraft]:
    out: list[RecommendationDraft] = []
    for r in sorted(records, key=lambda x: x.opportunity_score, reverse=True)[:6]:
        if r.intent == "comparison":
            ctype, title = "comparison", f"{r.query_text.title()}: an honest comparison"
        elif r.intent == "brand":
            ctype, title = "landing_page", f"{r.query_text.title()} — overview, pricing and FAQs"
        elif r.intent == "informational":
            ctype, title = "guide", f"The complete guide to {r.query_text}"
        else:
            ctype, title = "blog_post", f"Best {r.query_text} in 2026 (ranked)"
        gap = "not visible in organic results" if r.domain_visible is False else "not cited in AI answers"
        priority = "high" if r.opportunity_score >= 0.6 else "medium" if r.opportunity_score >= 0.4 else "low"
        out.append(
            RecommendationDraft(
                target_query_text=r.query_text,
                content_type=ctype,  # type: ignore[arg-type]
                title=title[:200],
                rationale=(
                    f"Opportunity score {r.opportunity_score}: brand is {gap}; "
                    f"difficulty {r.competitive_difficulty}/100."
                ),
                target_keywords=list(dict.fromkeys([r.query_text, *r.query_text.split()[:3]]))[:6],
                priority=priority,  # type: ignore[arg-type]
            )
        )
    return out


def analyze_fallback(state: RunState, deps: Deps) -> dict[str, Any]:
    records = state.get("records", [])
    visible = sum(1 for r in records if r.domain_visible)
    cited = sum(1 for r in records if r.ai_cited)
    analysis = AnalysisResult(
        executive_summary=(
            f"Rule-based analysis (LLM unavailable). {state['profile'].name} ranks for {visible}/{len(records)} "
            f"tracked queries and is cited in AI answers for {cited}/{len(records)}. "
            "Top opportunities are listed below."
        ),
        insights=_rule_insights(records)[:8],
        recommendations=_rule_recommendations(records),
    )
    return {
        "analysis": analysis,
        "analysis_source": "fallback",
        "degradation": ["analysis_fallback: rule-based insights"],
    }
