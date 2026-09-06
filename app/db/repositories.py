"""Persistence operations. Plain functions over a Session; no business logic beyond storage rules."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.dataforseo.domains import normalize_domain
from app.db.models import NodeExecution, PipelineRun, Profile, Query, Recommendation
from app.domain.models import NodeTrace, QueryRecord, RecommendationDraft, now_utc


# ── profiles ──────────────────────────────────────────────────────────────────
def create_profile(
    session: Session, *, name: str, domain: str, industry: str, description: str, competitors: list[str]
) -> Profile:
    profile = Profile(
        name=name.strip(),
        domain=normalize_domain(domain),
        industry=industry.strip(),
        description=(description or "").strip(),
        competitors=[normalize_domain(c) for c in competitors],
    )
    session.add(profile)
    session.flush()
    return profile


def get_profile(session: Session, profile_id: str) -> Profile | None:
    return session.get(Profile, profile_id)


def latest_full_run(session: Session, profile_id: str) -> PipelineRun | None:
    stmt = (
        select(PipelineRun)
        .where(PipelineRun.profile_id == profile_id, PipelineRun.mode == "full")
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def latest_run(session: Session, profile_id: str) -> PipelineRun | None:
    stmt = (
        select(PipelineRun).where(PipelineRun.profile_id == profile_id).order_by(PipelineRun.started_at.desc()).limit(1)
    )
    return session.scalars(stmt).first()


def profile_stats(session: Session, profile_id: str) -> dict[str, Any]:
    total_runs = session.scalar(select(func.count(PipelineRun.id)).where(PipelineRun.profile_id == profile_id)) or 0
    last = latest_run(session, profile_id)
    avg_score = session.scalar(select(func.avg(Query.opportunity_score)).where(Query.profile_id == profile_id))
    tracked = session.scalar(select(func.count(Query.id)).where(Query.profile_id == profile_id)) or 0
    visible = (
        session.scalar(
            select(func.count(Query.id)).where(Query.profile_id == profile_id, Query.domain_visible.is_(True))
        )
        or 0
    )
    return {
        "total_runs": int(total_runs),
        "last_run_uuid": last.id if last else None,
        "last_run_status": last.status if last else None,
        "last_run_at": last.started_at if last else None,
        "average_opportunity_score": round(float(avg_score), 4) if avg_score is not None else None,
        "tracked_queries": int(tracked),
        "visible_queries": int(visible),
    }


# ── runs ──────────────────────────────────────────────────────────────────────
def create_run(
    session: Session, profile_id: str, *, mode: str, question: str, recheck_query_id: str | None = None
) -> PipelineRun:
    run = PipelineRun(
        profile_id=profile_id, mode=mode, question=question, status="running", recheck_query_id=recheck_query_id
    )
    session.add(run)
    session.flush()
    return run


def get_run(session: Session, run_id: str) -> PipelineRun | None:
    return session.get(PipelineRun, run_id)


def list_runs(session: Session, profile_id: str, limit: int = 20) -> list[PipelineRun]:
    stmt = (
        select(PipelineRun)
        .where(PipelineRun.profile_id == profile_id)
        .order_by(PipelineRun.started_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def finish_run(
    session: Session,
    run: PipelineRun,
    *,
    status: str,
    planned_calls: int,
    records: int,
    api_calls: int,
    llm_calls: int,
    total_tokens: int,
    total_cost_usd: float,
    report_json: dict[str, Any] | None,
    summary: str | None,
    degradation: list[str],
    error: str | None = None,
) -> PipelineRun:
    finished = now_utc()
    run.status = status
    run.planned_calls_count = planned_calls
    run.records_extracted_count = records
    run.api_calls_count = api_calls
    run.llm_calls_count = llm_calls
    run.total_tokens = total_tokens
    run.total_cost_usd = total_cost_usd
    run.report_json = report_json
    run.summary = summary
    run.degradation = degradation
    run.error = error
    run.finished_at = finished
    started = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=finished.tzinfo)
    run.duration_ms = int((finished - started).total_seconds() * 1000)
    session.flush()
    return run


# ── queries ───────────────────────────────────────────────────────────────────
def upsert_queries(session: Session, profile_id: str, run_id: str, records: list[QueryRecord]) -> dict[str, Query]:
    existing = {q.query_text: q for q in session.scalars(select(Query).where(Query.profile_id == profile_id))}
    out: dict[str, Query] = {}
    now = now_utc()
    for rec in records:
        q = existing.get(rec.query_text) or Query(profile_id=profile_id, query_text=rec.query_text, discovered_at=now)
        q.last_run_id = run_id
        q.intent = rec.intent
        q.estimated_search_volume = rec.estimated_search_volume
        q.competition_index = rec.competition_index
        q.competitive_difficulty = rec.competitive_difficulty
        q.opportunity_score = rec.opportunity_score
        q.score_components = rec.score_components.model_dump() if rec.score_components else None
        q.domain_visible = rec.domain_visible
        q.visibility_status = rec.visibility_status
        q.visibility_position = rec.visibility_position
        q.ai_overview_present = rec.ai_overview_present
        q.ai_cited = rec.ai_cited
        q.competitor_positions = rec.competitor_positions
        q.competitors_cited = rec.competitors_cited
        q.top_domains = rec.top_domains
        q.sources = rec.sources
        q.error_flag = bool(rec.error_flags)
        q.error_detail = "; ".join(rec.error_flags) or None
        q.last_checked_at = now
        session.add(q)
        out[rec.query_text] = q
    session.flush()
    return out


def get_query(session: Session, query_id: str) -> Query | None:
    return session.get(Query, query_id)


def _current_run_ids(session: Session, profile_id: str):
    """Runs that define the profile's *current* picture: the latest full run plus any rechecks since."""
    latest = latest_full_run(session, profile_id)
    if latest is None:
        return select(PipelineRun.id).where(PipelineRun.profile_id == profile_id)
    return select(PipelineRun.id).where(
        PipelineRun.profile_id == profile_id,
        or_(
            PipelineRun.id == latest.id, (PipelineRun.mode == "recheck") & (PipelineRun.started_at >= latest.started_at)
        ),
    )


def list_queries(
    session: Session,
    profile_id: str,
    *,
    min_score: float | None = None,
    status: str | None = None,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[Query], int]:
    base = select(Query).where(
        Query.profile_id == profile_id, Query.last_run_id.in_(_current_run_ids(session, profile_id))
    )
    if min_score is not None:
        base = base.where(Query.opportunity_score >= min_score)
    if status:
        base = base.where(Query.visibility_status == status)
    total = session.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = session.scalars(
        base.order_by(Query.opportunity_score.desc(), Query.query_text).offset((page - 1) * per_page).limit(per_page)
    )
    return list(rows), int(total)


# ── recommendations ───────────────────────────────────────────────────────────
def replace_recommendations(
    session: Session,
    profile_id: str,
    run_id: str,
    drafts: list[RecommendationDraft],
    query_map: dict[str, Query],
    *,
    only_query_texts: set[str] | None = None,
) -> list[Recommendation]:
    """Full run: replace the whole set. Recheck: replace only those targeting the rechecked query."""
    stmt = select(Recommendation).where(Recommendation.profile_id == profile_id)
    if only_query_texts is not None:
        target_ids = [q.id for t, q in query_map.items() if t in only_query_texts]
        stmt = stmt.where(Recommendation.target_query_id.in_(target_ids))
    for old in session.scalars(stmt):
        session.delete(old)
    session.flush()

    out: list[Recommendation] = []
    for d in drafts:
        if only_query_texts is not None and d.target_query_text not in only_query_texts:
            continue
        target = query_map.get(d.target_query_text)
        rec = Recommendation(
            profile_id=profile_id,
            run_id=run_id,
            target_query_id=target.id if target else None,
            content_type=d.content_type,
            title=d.title,
            rationale=d.rationale,
            target_keywords=d.target_keywords,
            priority=d.priority,
        )
        session.add(rec)
        out.append(rec)
    session.flush()
    return out


def list_recommendations(session: Session, profile_id: str) -> list[Recommendation]:
    order = {"high": 0, "medium": 1, "low": 2}
    stmt = select(Recommendation).where(
        Recommendation.profile_id == profile_id, Recommendation.run_id.in_(_current_run_ids(session, profile_id))
    )
    rows = list(session.scalars(stmt))
    rows.sort(key=lambda r: (order.get(r.priority, 9), r.created_at))
    return rows


def recommendations_for_query(session: Session, query_id: str) -> list[Recommendation]:
    return list(session.scalars(select(Recommendation).where(Recommendation.target_query_id == query_id)))


# ── traces ────────────────────────────────────────────────────────────────────
def save_node_executions(session: Session, run_id: str, traces: list[NodeTrace]) -> None:
    for seq, t in enumerate(sorted(traces, key=lambda x: x.started_at)):
        session.add(
            NodeExecution(
                run_id=run_id,
                seq=seq,
                node_name=t.node,
                branch_key=t.branch_key,
                status=t.status,
                duration_ms=t.duration_ms,
                retry_count=t.retry_count,
                api_calls=t.api_calls,
                tokens=t.tokens,
                input_summary=t.input_summary,
                output_summary=t.output_summary,
                error=t.error,
                started_at=t.started_at,
            )
        )
    session.flush()


def list_node_executions(session: Session, run_id: str) -> list[NodeExecution]:
    stmt = select(NodeExecution).where(NodeExecution.run_id == run_id).order_by(NodeExecution.seq)
    return list(session.scalars(stmt))


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=now_utc().tzinfo)
