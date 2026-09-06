"""Extraction/Normalization agent: raw payloads → QueryRecord (deterministic, no LLM)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.dataforseo.parsers import ParseError, parse_keyword_volume, parse_llm_responses, parse_serp_organic
from app.domain.models import QueryRecord, RawResult, RunError
from app.graph.deps import Deps
from app.graph.state import RunState
from app.services.scoring import competitive_difficulty, opportunity_score


def normalize(state: RunState, deps: Deps) -> dict[str, Any]:
    profile = state["profile"]
    plan = state.get("plan")
    raw_results = state.get("raw_results", [])
    errors: list[RunError] = []

    # keyword volume: one batched call → dict by keyword
    volumes: dict[str, tuple[int | None, int | None]] = {}
    volume_failed = False
    for r in raw_results:
        if r.tool != "keyword_volume":
            continue
        if r.status != "ok" or r.payload is None:
            volume_failed = True
            continue
        try:
            volumes.update(parse_keyword_volume(r.payload).by_keyword)
        except ParseError as exc:
            volume_failed = True
            errors.append(RunError(node="normalize", kind="parse_error", message=f"keyword_volume: {exc}"))

    by_query: dict[str, list[RawResult]] = defaultdict(list)
    for r in raw_results:
        if r.tool != "keyword_volume":
            by_query[r.query_text].append(r)

    intents = {sq.query_text: sq.intent for sq in (plan.sub_queries if plan else [])}
    ordered_queries = [sq.query_text for sq in plan.sub_queries] if plan else list(by_query)

    records: list[QueryRecord] = []
    dropped = 0
    for query_text in ordered_queries:
        rec, ok_sources = _build_record(
            query_text, by_query.get(query_text, []), volumes, volume_failed, profile, errors
        )
        rec.intent = intents.get(query_text)
        if ok_sources == 0:
            dropped += 1
            continue
        records.append(rec)

    update: dict[str, Any] = {"records": records, "dropped_records": dropped}
    if errors:
        update["errors"] = errors
    if dropped:
        update["degradation"] = [f"normalize: dropped {dropped} query record(s) with no usable data"]
    return update


def _build_record(
    query_text: str,
    raws: list[RawResult],
    volumes: dict[str, tuple[int | None, int | None]],
    volume_failed: bool,
    profile: Any,
    errors: list[RunError],
) -> tuple[QueryRecord, int]:
    flags: list[str] = []
    sources: list[str] = []
    ok_sources = 0
    domain_visible: bool | None = None
    position: int | None = None
    ai_overview_present: bool | None = None
    ai_cited: bool | None = None
    competitor_positions: dict[str, int] = {}
    competitors_cited: list[str] = []
    top_domains: list[str] = []

    for r in raws:
        if r.status != "ok" or r.payload is None:
            flags.append(f"{r.tool}_failed:{r.error.kind if r.error else 'unknown'}")
            continue
        try:
            if r.tool == "serp_organic":
                s = parse_serp_organic(r.payload, profile_domain=profile.domain, competitors=profile.competitors)
                domain_visible = s.domain_position is not None
                position = s.domain_position
                competitor_positions = s.competitor_positions
                top_domains = s.top_domains
                ai_overview_present = s.ai_overview_present
                if s.ai_overview_present:
                    ai_cited = s.ai_overview_cites_domain or (ai_cited or False)
            elif r.tool == "llm_responses":
                lm = parse_llm_responses(r.payload, profile_domain=profile.domain, competitors=profile.competitors)
                ai_cited = bool(lm.domain_cited) if ai_cited is None else (ai_cited or lm.domain_cited)
                competitors_cited = lm.competitors_cited
            sources.append(r.tool)
            ok_sources += 1
        except ParseError as exc:
            flags.append(f"{r.tool}_parse_error")
            errors.append(
                RunError(node="normalize", kind="parse_error", message=f"{r.tool}: {exc}", query_text=query_text)
            )

    vol, comp_idx = volumes.get(query_text.lower(), (None, None))
    if vol is not None or comp_idx is not None:
        sources.append("keyword_volume")
        ok_sources += 1
    elif volume_failed:
        flags.append("keyword_volume_failed")

    difficulty = competitive_difficulty(comp_idx, len(competitor_positions))
    score, comps = opportunity_score(
        search_volume=vol, difficulty=difficulty, domain_visible=domain_visible, position=position, ai_cited=ai_cited
    )
    status = "unknown" if domain_visible is None else ("visible" if domain_visible else "not_visible")
    rec = QueryRecord(
        query_text=query_text,
        estimated_search_volume=vol,
        competition_index=comp_idx,
        competitive_difficulty=difficulty,
        opportunity_score=score,
        score_components=comps,
        domain_visible=domain_visible,
        visibility_status=status,
        visibility_position=position,
        ai_overview_present=ai_overview_present,
        ai_cited=ai_cited,
        competitor_positions=competitor_positions,
        competitors_cited=competitors_cited,
        top_domains=top_domains,
        sources=sources,
        error_flags=flags,
    )
    return rec, ok_sources


def normalize_validator(state: RunState, deps: Deps) -> dict[str, Any]:
    ok = len(state.get("records", [])) > 0
    update: dict[str, Any] = {"normalize_ok": ok}
    if not ok:
        update["degradation"] = ["normalize_validator: zero usable records → degraded report"]
    return update


def route_after_normalize_validation(state: RunState) -> str:
    return "analyze" if state.get("normalize_ok") else "degraded_report"
