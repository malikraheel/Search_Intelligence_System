"""`traced_node` wraps every graph node with timing, structured logging, metrics and a NodeTrace.

Nodes may return a private `_trace` dict (retry_count, api_calls, tokens, status) which is
consumed here and never written to graph state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.domain.models import NodeTrace
from app.observability.logging import bind_context, get_logger, redact
from app.observability.metrics import MetricsRegistry

log = get_logger("graph")

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

_INPUT_KEYS = ("mode", "question", "recheck_query", "plan_source", "plan_errors", "attempt", "validation_errors")


def _summarize_input(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: state[k] for k in _INPUT_KEYS if k in state}
    if (profile := state.get("profile")) is not None:
        out["profile"] = {"domain": profile.domain, "competitors": len(profile.competitors)}
    if (pc := state.get("planned_call")) is not None:
        out["planned_call"] = {"tool": pc.tool, "query_text": pc.query_text}
    if (plan := state.get("plan")) is not None:
        out["plan_queries"] = len(plan.sub_queries)
    for key in ("planned_calls", "raw_results", "records"):
        if key in state:
            out[f"{key}_count"] = len(state[key])
    return redact(out)


def _summarize_output(update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in update.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            out[f"{key}_count"] = len(value)
            if key == "validation_errors" and value:
                out[key] = value[:5]
        elif isinstance(value, str | int | float | bool) or value is None:
            out[key] = value
        elif hasattr(value, "model_dump"):
            out[key] = type(value).__name__
        else:
            out[key] = type(value).__name__
    return redact(out)


def traced_node(name: str, metrics: MetricsRegistry) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            pc = state.get("planned_call")
            branch_key = pc.call_id[:8] if pc is not None else None
            ctx: dict[str, Any] = {"node": name, "run_id": state.get("run_id")}
            if branch_key:
                ctx["branch"] = branch_key
            started = time.perf_counter()
            with bind_context(**ctx):
                log.info("node.start", input=_summarize_input(state))
                try:
                    update = fn(state) or {}
                except Exception as exc:
                    duration = (time.perf_counter() - started) * 1000
                    metrics.record_node(name, duration, ok=False)
                    log.error("node.end", status="failure", duration_ms=round(duration, 1), error=repr(exc))
                    raise
                duration = (time.perf_counter() - started) * 1000
                hint = update.pop("_trace", {}) or {}
                status = hint.get("status") or ("degraded" if update.get("errors") else "success")
                trace = NodeTrace(
                    node=name,
                    branch_key=branch_key,
                    status=status,
                    duration_ms=round(duration, 1),
                    retry_count=int(hint.get("retry_count", 0)),
                    api_calls=int(hint.get("api_calls", 0)),
                    tokens=int(hint.get("tokens", sum(u.total_tokens for u in update.get("llm_usage", []) or []))),
                    input_summary=_summarize_input(state),
                    output_summary=_summarize_output(update),
                    error=hint.get("error"),
                )
                metrics.record_node(name, duration, ok=status != "failure")
                log.info(
                    "node.end",
                    status=status,
                    duration_ms=trace.duration_ms,
                    retry_count=trace.retry_count,
                    api_calls=trace.api_calls,
                    tokens=trace.tokens,
                    output=trace.output_summary,
                    error=trace.error,
                )
                update["node_traces"] = [*update.get("node_traces", []), trace]
                return update

        wrapper.__name__ = name
        return wrapper

    return decorator
