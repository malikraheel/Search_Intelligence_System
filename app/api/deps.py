from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.services.pipeline_service import PipelineService


def get_session(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_pipeline(request: Request) -> PipelineService:
    return request.app.state.pipeline


def get_metrics(request: Request):
    return request.app.state.deps.metrics


def get_gateway(request: Request):
    return request.app.state.deps.gateway
