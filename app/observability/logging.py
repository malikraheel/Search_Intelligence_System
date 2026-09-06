"""Structured JSON logging with correlation context and secret redaction.

Every log line carries whatever has been bound via `bind_context` (trace_id, run_id,
node, ...) through contextvars, so a single DAG run can be followed node-by-node.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import structlog

# Matches credential-like keys (api_key, password, authorization, access_token, login, ...) but NOT
# token *counts* such as `tokens`, `total_tokens`, `prompt_tokens`.
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|password|passwd|secret|authorization|credential|bearer|login"
    r"|(?:^|[_-])(?:auth|access|id|refresh|session)?[_-]?token$)",
    re.IGNORECASE,
)
_MAX_STR = 500
_REDACTED = "***REDACTED***"


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Recursively mask sensitive keys and truncate long strings. Safe on any JSON-ish value."""
    if _depth > 8:
        return "<max-depth>"
    if isinstance(value, Mapping):
        return {
            str(k): (_REDACTED if _SENSITIVE_KEY.search(str(k)) else redact(v, _depth=_depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [redact(v, _depth=_depth + 1) for v in list(value)[:50]]
    if isinstance(value, str) and len(value) > _MAX_STR:
        return value[:_MAX_STR] + f"…(+{len(value) - _MAX_STR} chars)"
    if hasattr(value, "model_dump"):
        return redact(value.model_dump(mode="json"), _depth=_depth + 1)
    return value


def _redact_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict.keys()):
        if key in ("event", "level", "timestamp", "logger"):
            continue
        event_dict[key] = _REDACTED if _SENSITIVE_KEY.search(key) else redact(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.dev.ConsoleRenderer() if fmt == "console" else structlog.processors.JSONRenderer(default=str)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
    # Quiet noisy third-party loggers; our own logs go through structlog.
    for name in ("httpx", "httpcore", "LiteLLM", "litellm", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    # NOTE: pass `component` as an initial value rather than calling `.bind()` here. `.bind()` on the
    # lazy proxy materialises a logger with the configuration active *at import time*, which would
    # freeze module-level loggers to structlog's default console renderer before configure_logging runs.
    return structlog.get_logger(component=name)


@contextmanager
def bind_context(**kwargs: Any) -> Iterator[None]:
    """Temporarily bind correlation fields (trace_id, run_id, node, ...) to all log lines."""
    tokens = structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


def bind_permanent(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
