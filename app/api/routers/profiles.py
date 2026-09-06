from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Query as QueryParam
from sqlalchemy.orm import Session

from app.api.deps import get_pipeline, get_session
from app.api.schemas import (
    Page,
    ProfileCreate,
    ProfileDetailOut,
    ProfileOut,
    ProfileStats,
    QueryOut,
    RecommendationOut,
    RunOut,
    RunRequest,
)
from app.api.serializers import profile_out, query_out, recommendation_out
from app.db import repositories as repo
from app.services.pipeline_service import PipelineService, run_to_dict

router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])


def _profile_or_404(session: Session, profile_uuid: str):
    profile = repo.get_profile(session, profile_uuid)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile '{profile_uuid}' not found")
    return profile


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProfileOut)
def create_profile(body: ProfileCreate, session: Session = Depends(get_session)) -> ProfileOut:
    profile = repo.create_profile(
        session,
        name=body.name,
        domain=body.domain,
        industry=body.industry,
        description=body.description,
        competitors=body.competitors,
    )
    session.commit()
    return profile_out(profile)


@router.get("/{profile_uuid}", response_model=ProfileDetailOut)
def get_profile(profile_uuid: str, session: Session = Depends(get_session)) -> ProfileDetailOut:
    profile = _profile_or_404(session, profile_uuid)
    stats = repo.profile_stats(session, profile.id)
    stats["last_run_at"] = repo.as_utc(stats["last_run_at"])
    return ProfileDetailOut(**profile_out(profile).model_dump(), stats=ProfileStats(**stats))


@router.post("/{profile_uuid}/run", response_model=RunOut)
def run_pipeline(
    profile_uuid: str,
    body: RunRequest | None = None,
    session: Session = Depends(get_session),
    pipeline: PipelineService = Depends(get_pipeline),
) -> RunOut:
    """Trigger the full DAG synchronously. Always 200: `status` says completed / partial / failed."""
    profile = _profile_or_404(session, profile_uuid)
    run = pipeline.run_profile(session, profile, question=body.question if body else None)
    return RunOut(**run_to_dict(run))


@router.get("/{profile_uuid}/runs", response_model=list[RunOut])
def list_runs(profile_uuid: str, limit: int = QueryParam(20, ge=1, le=100), session: Session = Depends(get_session)):
    profile = _profile_or_404(session, profile_uuid)
    return [RunOut(**run_to_dict(r)) for r in repo.list_runs(session, profile.id, limit=limit)]


@router.get("/{profile_uuid}/queries", response_model=Page[QueryOut])
def list_queries(
    profile_uuid: str,
    min_score: float | None = QueryParam(None, ge=0.0, le=1.0),
    status_: str | None = QueryParam(None, alias="status", pattern="^(visible|not_visible|unknown)$"),
    page: int = QueryParam(1, ge=1),
    per_page: int = QueryParam(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> Page[QueryOut]:
    profile = _profile_or_404(session, profile_uuid)
    rows, total = repo.list_queries(
        session, profile.id, min_score=min_score, status=status_, page=page, per_page=per_page
    )
    return Page[QueryOut](items=[query_out(q) for q in rows], page=page, per_page=per_page, total=total)


@router.get("/{profile_uuid}/recommendations", response_model=list[RecommendationOut])
def list_recommendations(profile_uuid: str, session: Session = Depends(get_session)) -> list[RecommendationOut]:
    profile = _profile_or_404(session, profile_uuid)
    return [recommendation_out(r) for r in repo.list_recommendations(session, profile.id)]
