"""ORM → response-schema mapping."""

from __future__ import annotations

from app.api.schemas import NodeExecutionOut, ProfileOut, QueryOut, RecommendationOut
from app.db import repositories as repo
from app.db.models import NodeExecution, Profile, Query, Recommendation


def profile_out(p: Profile) -> ProfileOut:
    return ProfileOut(
        profile_uuid=p.id,
        name=p.name,
        domain=p.domain,
        industry=p.industry,
        description=p.description or "",
        competitors=list(p.competitors or []),
        status=p.status,
        created_at=repo.as_utc(p.created_at),  # type: ignore[arg-type]
    )


def query_out(q: Query) -> QueryOut:
    return QueryOut(
        query_uuid=q.id,
        query_text=q.query_text,
        intent=q.intent,
        estimated_search_volume=q.estimated_search_volume,
        competitive_difficulty=q.competitive_difficulty,
        opportunity_score=q.opportunity_score,
        score_components=q.score_components,
        domain_visible=q.domain_visible,
        visibility_status=q.visibility_status,  # type: ignore[arg-type]
        visibility_position=q.visibility_position,
        ai_overview_present=q.ai_overview_present,
        ai_cited=q.ai_cited,
        competitor_positions=dict(q.competitor_positions or {}),
        competitors_cited=list(q.competitors_cited or []),
        error_flag=q.error_flag,
        error_detail=q.error_detail,
        discovered_at=repo.as_utc(q.discovered_at),  # type: ignore[arg-type]
        last_checked_at=repo.as_utc(q.last_checked_at),  # type: ignore[arg-type]
        last_run_uuid=q.last_run_id,
    )


def recommendation_out(r: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        recommendation_uuid=r.id,
        target_query_uuid=r.target_query_id,
        target_query_text=r.target_query.query_text if r.target_query else None,
        content_type=r.content_type,
        title=r.title,
        rationale=r.rationale,
        target_keywords=list(r.target_keywords or []),
        priority=r.priority,  # type: ignore[arg-type]
        run_uuid=r.run_id,
        created_at=repo.as_utc(r.created_at),  # type: ignore[arg-type]
    )


def node_execution_out(n: NodeExecution) -> NodeExecutionOut:
    return NodeExecutionOut(
        seq=n.seq,
        node_name=n.node_name,
        branch_key=n.branch_key,
        status=n.status,
        duration_ms=n.duration_ms,
        retry_count=n.retry_count,
        api_calls=n.api_calls,
        tokens=n.tokens,
        input_summary=n.input_summary or {},
        output_summary=n.output_summary or {},
        error=n.error,
        started_at=repo.as_utc(n.started_at),  # type: ignore[arg-type]
    )
