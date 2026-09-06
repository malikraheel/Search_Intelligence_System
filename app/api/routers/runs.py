from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_gateway, get_metrics, get_session
from app.api.schemas import RunOut, RunTraceOut
from app.api.serializers import node_execution_out
from app.db import repositories as repo
from app.services.pipeline_service import run_to_dict

router = APIRouter(prefix="/api/v1", tags=["runs", "observability"])


@router.get("/runs/{run_uuid}", response_model=RunOut)
def get_run(run_uuid: str, session: Session = Depends(get_session)) -> RunOut:
    run = repo.get_run(session, run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run '{run_uuid}' not found")
    return RunOut(**run_to_dict(run))


@router.get("/runs/{run_uuid}/trace", response_model=RunTraceOut)
def get_run_trace(run_uuid: str, session: Session = Depends(get_session)) -> RunTraceOut:
    """Node-by-node execution trace of a single run (the persisted correlation-ID trace log)."""
    run = repo.get_run(session, run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run '{run_uuid}' not found")
    nodes = [node_execution_out(n) for n in repo.list_node_executions(session, run.id)]
    return RunTraceOut(run_uuid=run.id, status=run.status, nodes=nodes)


@router.get("/metrics")
def metrics(registry=Depends(get_metrics), gateway=Depends(get_gateway)) -> dict[str, Any]:
    """In-memory metrics: per-node latency/success, API call counts, LLM usage, breaker states."""
    return {
        **registry.snapshot(),
        "llm_usage": gateway.ledger.snapshot(),
        "breakers": gateway.breakers.snapshot(),
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
