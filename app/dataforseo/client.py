"""DataForSEO client with pluggable transports (live / mock / chaos).

The client owns: timeouts, retry with backoff+jitter, circuit breaker per endpoint,
error classification (HTTP + task-level status codes), and API-call metrics.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from app.config import Settings
from app.dataforseo import mock_data
from app.observability.logging import get_logger
from app.observability.metrics import MetricsRegistry
from app.resilience import (
    BreakerRegistry,
    CircuitOpenError,
    NonRetryableError,
    RetryableError,
    RetryPolicy,
    call_with_retry,
)
from app.resilience.errors import raise_for_http_status
from app.tools.registry import ToolSpec

log = get_logger("dataforseo")


@dataclass
class CallContext:
    """Profile-derived context the mock transport uses to make results realistic."""

    profile_domain: str = "example.com"
    competitors: list[str] = field(default_factory=list)


class Transport(Protocol):
    def post(self, path: str, payload: list[dict[str, Any]], ctx: CallContext) -> dict[str, Any]: ...


# ── live ──────────────────────────────────────────────────────────────────────
class LiveTransport:
    def __init__(self, base_url: str, login: str, password: str, *, connect_timeout_s: float, read_timeout_s: float):
        if not login or not password:
            raise NonRetryableError("DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD are required in live mode", kind="config")
        self._client = httpx.Client(
            base_url=base_url,
            auth=httpx.BasicAuth(login, password),
            timeout=httpx.Timeout(connect=connect_timeout_s, read=read_timeout_s, write=10.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )

    def post(self, path: str, payload: list[dict[str, Any]], ctx: CallContext) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise RetryableError(f"timeout calling {path}: {exc}", kind="timeout") from exc
        except httpx.TransportError as exc:
            raise RetryableError(f"network error calling {path}: {exc}", kind="network") from exc
        raise_for_http_status(response.status_code, response.text)
        try:
            return response.json()
        except ValueError as exc:
            raise NonRetryableError(
                f"non-JSON body from {path}", kind="malformed", http_status=response.status_code
            ) from exc

    def close(self) -> None:
        self._client.close()


# ── mock ──────────────────────────────────────────────────────────────────────
class MockTransport:
    def post(self, path: str, payload: list[dict[str, Any]], ctx: CallContext) -> dict[str, Any]:
        task = payload[0]
        if path.endswith("/serp/google/organic/live/advanced"):
            return mock_data.serp_organic_response(
                keyword=task["keyword"],
                location_code=task.get("location_code", 2840),
                language_code=task.get("language_code", "en"),
                depth=task.get("depth", 10),
                profile_domain=ctx.profile_domain,
                competitors=ctx.competitors,
            )
        if path.endswith("/llm_responses/live"):
            return mock_data.llm_responses_response(
                prompt=task["user_prompt"],
                model_name=task.get("model_name", "gpt-4o-mini"),
                profile_domain=ctx.profile_domain,
                competitors=ctx.competitors,
            )
        if path.endswith("/search_volume/live"):
            return mock_data.keyword_volume_response(
                keywords=task["keywords"],
                location_code=task.get("location_code", 2840),
                language_code=task.get("language_code", "en"),
            )
        raise NonRetryableError(f"mock transport has no fixture for {path}", kind="mock", http_status=404)


# ── chaos ─────────────────────────────────────────────────────────────────────
_PATH_TO_TOOL = {
    "/serp/google/organic/live/advanced": "serp_organic",
    "/llm_responses/live": "llm_responses",
    "/search_volume/live": "keyword_volume",
}


def parse_chaos_script(script: str) -> dict[str, list[str]]:
    """'serp_organic:429,429,ok;llm_responses:500,500' → {'serp_organic': ['429','429','ok'], ...}"""
    out: dict[str, list[str]] = {}
    for chunk in filter(None, (c.strip() for c in script.split(";"))):
        tool, _, seq = chunk.partition(":")
        out[tool.strip()] = [s.strip().lower() for s in seq.split(",") if s.strip()]
    return out


class ChaosTransport:
    """Wraps another transport and injects a scripted failure sequence per tool, then passes through."""

    def __init__(self, inner: Transport, script: str | dict[str, list[str]]):
        self._inner = inner
        self._script = (
            parse_chaos_script(script) if isinstance(script, str) else {k: list(v) for k, v in script.items()}
        )
        self._lock = threading.Lock()

    def _next_outcome(self, tool: str) -> str:
        with self._lock:
            seq = self._script.get(tool)
            if seq:
                return seq.pop(0)
        return "ok"

    def post(self, path: str, payload: list[dict[str, Any]], ctx: CallContext) -> dict[str, Any]:
        tool = next((t for suffix, t in _PATH_TO_TOOL.items() if path.endswith(suffix)), "unknown")
        outcome = self._next_outcome(tool)
        if outcome == "ok":
            return self._inner.post(path, payload, ctx)
        if outcome == "timeout":
            raise RetryableError(f"chaos: simulated timeout for {tool}", kind="timeout")
        if outcome == "network":
            raise RetryableError(f"chaos: simulated connection reset for {tool}", kind="network")
        if outcome == "malformed":
            return {"status_code": 20000, "tasks": [{"status_code": 20000, "result": [{"items": "not-a-list"}]}]}
        if outcome.isdigit():
            raise_for_http_status(int(outcome), f"chaos: simulated HTTP {outcome} for {tool}")
        raise NonRetryableError(f"chaos: unknown outcome '{outcome}'", kind="chaos")


# ── client ────────────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    payload: dict[str, Any]
    attempts: int
    latency_ms: float

    @property
    def retry_count(self) -> int:
        return max(self.attempts - 1, 0)


def _check_task_status(body: dict[str, Any], path: str) -> None:
    """DataForSEO returns HTTP 200 with per-task status codes; classify them."""
    top = body.get("status_code")
    if top is not None and top != 20000:
        msg = f"DataForSEO status {top}: {body.get('status_message', '')}"
        if 50000 <= int(top) < 60000 or int(top) in (40202, 40210):
            raise RetryableError(msg, kind="api_status", api_status_code=int(top))
        raise NonRetryableError(msg, kind="api_status", api_status_code=int(top))
    tasks = body.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise NonRetryableError(f"malformed response from {path}: no tasks", kind="malformed")
    code = tasks[0].get("status_code")
    if code == 20000:
        return
    msg = f"DataForSEO task status {code}: {tasks[0].get('status_message', '')}"
    if code is None:
        raise NonRetryableError(f"malformed response from {path}: task without status_code", kind="malformed")
    if 50000 <= int(code) < 60000 or int(code) in (40202, 40210):  # 402xx: rate/limit → retryable
        raise RetryableError(msg, kind="api_status", api_status_code=int(code))
    raise NonRetryableError(msg, kind="api_status", api_status_code=int(code))


class DataForSEOClient:
    def __init__(
        self,
        transport: Transport,
        *,
        policy: RetryPolicy,
        breakers: BreakerRegistry,
        metrics: MetricsRegistry,
    ):
        self._transport = transport
        self._policy = policy
        self._breakers = breakers
        self._metrics = metrics

    def call(self, spec: ToolSpec, inp: BaseModel, ctx: CallContext | None = None) -> ToolResult:
        """Execute one validated tool call with retry/backoff + breaker. Raises AppError on failure."""
        ctx = ctx or CallContext()
        payload = [inp.model_dump(mode="json")]
        breaker = self._breakers.get(spec.name)
        started = time.perf_counter()

        def _once() -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                body = self._transport.post(spec.endpoint_path, payload, ctx)
                _check_task_status(body, spec.endpoint_path)
            except RetryableError as exc:
                self._metrics.record_api_call(spec.name, "retryable_error")
                log.warning(
                    "dataforseo.call_failed",
                    tool=spec.name,
                    retryable=True,
                    error=str(exc),
                    http_status=exc.http_status,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                raise
            except NonRetryableError as exc:
                self._metrics.record_api_call(spec.name, "non_retryable_error")
                log.error(
                    "dataforseo.call_failed",
                    tool=spec.name,
                    retryable=False,
                    error=str(exc),
                    http_status=exc.http_status,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                raise
            self._metrics.record_api_call(spec.name, "ok")
            return body

        def _on_retry(attempt: int, exc: BaseException, sleep_s: float) -> None:
            log.warning("dataforseo.retry", tool=spec.name, attempt=attempt, sleep_s=round(sleep_s, 2), error=str(exc))

        try:
            outcome = call_with_retry(_once, policy=self._policy, breaker=breaker, on_retry=_on_retry)
        except CircuitOpenError:
            self._metrics.record_api_call(spec.name, "breaker_open")
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        return ToolResult(payload=outcome.value, attempts=outcome.attempts, latency_ms=latency_ms)


def build_transport(settings: Settings) -> Transport:
    mock = MockTransport()
    if settings.dataforseo_mode == "mock":
        return mock
    if settings.dataforseo_mode == "chaos":
        return ChaosTransport(mock, settings.dataforseo_chaos_script)
    return LiveTransport(
        settings.dataforseo_base_url,
        settings.dataforseo_login or "",
        settings.dataforseo_password or "",
        connect_timeout_s=settings.dataforseo_connect_timeout_s,
        read_timeout_s=settings.dataforseo_read_timeout_s,
    )


def build_client(settings: Settings, metrics: MetricsRegistry, transport: Transport | None = None) -> DataForSEOClient:
    return DataForSEOClient(
        transport or build_transport(settings),
        policy=RetryPolicy(
            max_attempts=settings.dataforseo_max_attempts,
            base_delay_s=settings.dataforseo_retry_base_delay_s,
            max_delay_s=settings.dataforseo_retry_max_delay_s,
        ),
        breakers=BreakerRegistry(
            failure_threshold=settings.dataforseo_breaker_failure_threshold,
            reset_timeout_s=settings.dataforseo_breaker_reset_s,
        ),
        metrics=metrics,
    )
