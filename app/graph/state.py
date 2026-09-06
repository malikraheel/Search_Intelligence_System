"""Graph state schemas.

TypedDicts (not Pydantic) because LangGraph's `Send` passes branch payloads as plain dicts.
Every key that parallel branches write MUST carry a reducer, otherwise LangGraph raises
InvalidUpdateError when two branches finish in the same superstep.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from app.domain.models import (
    AnalysisResult,
    ApiCallRecord,
    ErrorInfo,
    LLMUsage,
    NodeTrace,
    PlannedCall,
    ProfileCtx,
    QueryRecord,
    RawResult,
    Report,
    RetrievalPlan,
    RunError,
)
from app.tools.schemas import ToolCallProposal

RunMode = Literal["full", "recheck"]
RunStatus = Literal["completed", "partial", "failed"]


class RunState(TypedDict, total=False):
    # ── inputs ──
    mode: RunMode
    run_id: str
    profile: ProfileCtx
    question: str
    recheck_query: str | None

    # ── planning ──
    plan: RetrievalPlan | None
    plan_errors: list[str]
    plan_repair_attempts: int
    plan_source: Literal["llm", "repaired", "fallback", "recheck"]
    planned_calls: list[PlannedCall]

    # ── fan-in channels (written by parallel branches → reducers) ──
    raw_results: Annotated[list[RawResult], operator.add]
    api_calls: Annotated[list[ApiCallRecord], operator.add]
    llm_usage: Annotated[list[LLMUsage], operator.add]
    node_traces: Annotated[list[NodeTrace], operator.add]
    errors: Annotated[list[RunError], operator.add]
    degradation: Annotated[list[str], operator.add]

    # ── downstream ──
    records: list[QueryRecord]
    dropped_records: int
    normalize_ok: bool
    analysis: AnalysisResult | None
    analysis_source: Literal["llm", "fallback"]
    report: Report | None
    status: RunStatus


class BranchInput(TypedDict):
    run_id: str
    profile: ProfileCtx
    planned_call: PlannedCall


class BranchState(TypedDict, total=False):
    # input
    run_id: str
    profile: ProfileCtx
    planned_call: PlannedCall
    # working
    proposal: ToolCallProposal | None
    validation_errors: list[str]
    attempt: int
    llm_unavailable: bool
    validated_tool: str | None
    validated_args: dict[str, Any] | None
    exec_ok: bool
    exec_error: ErrorInfo | None
    exec_retry_count: int
    # shared with the parent (same reducers)
    raw_results: Annotated[list[RawResult], operator.add]
    api_calls: Annotated[list[ApiCallRecord], operator.add]
    llm_usage: Annotated[list[LLMUsage], operator.add]
    node_traces: Annotated[list[NodeTrace], operator.add]
    errors: Annotated[list[RunError], operator.add]
    degradation: Annotated[list[str], operator.add]


class BranchOutput(TypedDict, total=False):
    """Only reducer-backed keys leave the subgraph, so N parallel branches merge cleanly."""

    raw_results: Annotated[list[RawResult], operator.add]
    api_calls: Annotated[list[ApiCallRecord], operator.add]
    llm_usage: Annotated[list[LLMUsage], operator.add]
    node_traces: Annotated[list[NodeTrace], operator.add]
    errors: Annotated[list[RunError], operator.add]
    degradation: Annotated[list[str], operator.add]
