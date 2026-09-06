from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_pipeline, get_session
from app.api.schemas import QueryOut, RecheckOut, RunOut
from app.api.serializers import query_out, recommendation_out
from app.db import repositories as repo
from app.services.pipeline_service import PipelineService, run_to_dict

router = APIRouter(prefix="/api/v1/queries", tags=["queries"])


def _query_or_404(session: Session, query_uuid: str):
    query = repo.get_query(session, query_uuid)
    if query is None:
        raise HTTPException(status_code=404, detail=f"query '{query_uuid}' not found")
    return query


@router.get("/{query_uuid}", response_model=QueryOut)
def get_query(query_uuid: str, session: Session = Depends(get_session)) -> QueryOut:
    return query_out(_query_or_404(session, query_uuid))


@router.post("/{query_uuid}/recheck", response_model=RecheckOut)
def recheck_query(
    query_uuid: str,
    session: Session = Depends(get_session),
    pipeline: PipelineService = Depends(get_pipeline),
) -> RecheckOut:
    """Re-run Retrieval → Normalization → Analysis for ONE query and return the updated record."""
    query = _query_or_404(session, query_uuid)
    profile = repo.get_profile(session, query.profile_id)
    if profile is None:  # pragma: no cover – FK guarantees this
        raise HTTPException(status_code=404, detail="profile not found")
    run = pipeline.recheck_query(session, profile, query)
    session.refresh(query)
    return RecheckOut(
        run=RunOut(**run_to_dict(run)),
        query=query_out(query),
        recommendations=[recommendation_out(r) for r in repo.recommendations_for_query(session, query.id)],
    )
