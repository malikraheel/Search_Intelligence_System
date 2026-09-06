"""Retrieval branch (compiled as a subgraph; one instance per planned call):

retrieval_agent (LLM proposes a tool call)
  → tool_arg_validator (code: Pydantic-validate; repair once; else fallback)
    → execute_tool (code: DataForSEO call with retry/backoff/breaker)
      → [ok] END  |  [failed] retrieval_fallback (code: degraded record with error flag) → END
"""

from __future__ import annotations

from typing import Any

from app.dataforseo.client import CallContext
from app.domain.models import ApiCallRecord, ErrorInfo, RawResult, RunError
from app.gateway import AllProvidersFailed, Role
from app.graph.deps import Deps
from app.graph.prompts import retrieval_messages
from app.graph.state import BranchState
from app.resilience.errors import AppError
from app.tools.registry import TOOLS, to_openai_tools, validate_tool_call

MAX_TOOL_ARG_ATTEMPTS = 2


def retrieval_agent(state: BranchState, deps: Deps) -> dict[str, Any]:
    planned = state["planned_call"]
    attempt = state.get("attempt", 0) + 1
    messages = retrieval_messages(
        state["profile"],
        planned,
        deps.settings.pipeline_default_location_code,
        deps.settings.pipeline_default_language_code,
        repair_errors=state.get("validation_errors") if attempt > 1 else None,
    )
    try:
        resp = deps.gateway.complete(
            role=Role.RETRIEVAL,
            messages=messages,
            run_id=state["run_id"],
            tools=to_openai_tools(),  # all tools exposed; the LLM picks (validator checks it matches the plan)
            tool_choice="auto",
            use_cache=attempt == 1,
        )
    except AllProvidersFailed as exc:
        return {
            "proposal": None,
            "attempt": attempt,
            "llm_unavailable": True,
            "validation_errors": [f"retrieval LLM unavailable: {exc}"],
            "errors": [
                RunError(node="retrieval_agent", kind=exc.kind, message=str(exc), query_text=planned.query_text)
            ],
        }
    proposal = resp.tool_calls[0] if resp.tool_calls else None
    return {"proposal": proposal, "attempt": attempt, "llm_unavailable": False, "llm_usage": [resp.usage]}


def _defaults_for(state: BranchState, deps: Deps) -> dict[str, Any]:
    planned = state["planned_call"]
    defaults: dict[str, Any] = {
        "location_code": deps.settings.pipeline_default_location_code,
        "language_code": deps.settings.pipeline_default_language_code,
    }
    # Fill the ONE argument we know for certain from the plan if the LLM omitted it (partial call).
    if planned.tool == "serp_organic":
        defaults["keyword"] = planned.query_text
    elif planned.tool == "llm_responses":
        defaults["user_prompt"] = f"what is the best option for {planned.query_text}?"
    elif planned.tool == "keyword_volume":
        defaults["keywords"] = planned.keywords
    return defaults


def tool_arg_validator(state: BranchState, deps: Deps) -> dict[str, Any]:
    planned = state["planned_call"]
    proposal = state.get("proposal")
    if proposal is None:
        errs = state.get("validation_errors") or ["no tool call was proposed"]
        return {"validation_errors": errs, "validated_tool": None, "validated_args": None}

    errors: list[str] = []
    if proposal.name != planned.tool:
        errors.append(f"expected tool '{planned.tool}' for this step but got '{proposal.name}'")
    raw_args = proposal.arguments if proposal.arguments is not None else proposal.raw_arguments
    outcome = validate_tool_call(proposal.name, raw_args, defaults=_defaults_for(state, deps))
    if proposal.parse_error and not outcome.ok:
        errors.append(proposal.parse_error)
    errors.extend(outcome.errors)

    if errors or not outcome.ok or outcome.input is None:
        return {"validation_errors": errors, "validated_tool": None, "validated_args": None}
    return {
        "validation_errors": [],
        "validated_tool": proposal.name,
        "validated_args": outcome.input.model_dump(mode="json"),
    }


def route_after_validation(state: BranchState) -> str:
    if state.get("validated_args"):
        return "execute_tool"
    if state.get("attempt", 0) < MAX_TOOL_ARG_ATTEMPTS and not state.get("llm_unavailable"):
        return "retrieval_agent"
    return "retrieval_fallback"


def execute_tool(state: BranchState, deps: Deps) -> dict[str, Any]:
    planned = state["planned_call"]
    tool_name = state["validated_tool"]
    assert tool_name is not None and state.get("validated_args") is not None
    spec = TOOLS[tool_name]
    inp = spec.input_model.model_validate(state["validated_args"])
    profile = state["profile"]
    ctx = CallContext(profile_domain=profile.domain, competitors=profile.competitors)
    try:
        result = deps.dfs_client.call(spec, inp, ctx)
    except AppError as exc:
        info = ErrorInfo(
            kind=exc.kind,
            message=exc.message,
            http_status=exc.http_status,
            api_status_code=exc.api_status_code,
            retryable=exc.retryable,
        )
        return {
            "exec_ok": False,
            "exec_error": info,
            "api_calls": [ApiCallRecord(tool=tool_name, outcome=exc.kind, latency_ms=0.0, attempts=0)],
            "errors": [
                RunError(node="execute_tool", kind=exc.kind, message=exc.message, query_text=planned.query_text)
            ],
            "_trace": {"status": "degraded", "error": exc.message},
        }
    raw = RawResult(
        call_id=planned.call_id,
        tool=tool_name,  # type: ignore[arg-type]
        query_text=planned.query_text,
        arguments=state["validated_args"] or {},
        status="ok",
        payload=result.payload,
        latency_ms=round(result.latency_ms, 1),
        retry_count=result.retry_count,
    )
    return {
        "exec_ok": True,
        "exec_error": None,
        "exec_retry_count": result.retry_count,
        "raw_results": [raw],
        "api_calls": [
            ApiCallRecord(tool=tool_name, outcome="ok", latency_ms=result.latency_ms, attempts=result.attempts)
        ],
        "_trace": {"retry_count": result.retry_count, "api_calls": result.attempts},
    }


def route_after_execute(state: BranchState) -> str:
    return "done" if state.get("exec_ok") else "retrieval_fallback"


def retrieval_fallback(state: BranchState, deps: Deps) -> dict[str, Any]:
    """Graceful degradation: emit a clearly flagged empty result so downstream nodes still run."""
    planned = state["planned_call"]
    err = state.get("exec_error")
    if err is None:
        msgs = state.get("validation_errors") or ["unknown"]
        err = ErrorInfo(kind="tool_args_invalid", message="; ".join(msgs)[:500], retryable=False)
        reason = f"tool-call validation failed after {state.get('attempt', 0)} attempt(s)"
    else:
        reason = f"{planned.tool} failed ({err.kind}): {err.message[:200]}"
    raw = RawResult(
        call_id=planned.call_id,
        tool=planned.tool,
        query_text=planned.query_text,
        arguments=state.get("validated_args") or {},
        status="degraded",
        payload=None,
        error=err,
        fallback_reason=reason,
    )
    return {
        "raw_results": [raw],
        "degradation": [f"{planned.tool}[{planned.query_text}]: {reason}"],
        "_trace": {"status": "degraded", "error": reason},
    }
