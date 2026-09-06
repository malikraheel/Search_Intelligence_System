"""Scripted provider for tests and offline demos. Never touches the network."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from app.gateway.types import LLMRequest, ProviderResponse
from app.tools.schemas import ToolCallProposal

Scripted = ProviderResponse | Exception | Callable[[LLMRequest], ProviderResponse]


def text_response(
    text: str, *, tokens: int = 100, model: str = "fake/model", cost: float | None = 0.0001
) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        prompt_tokens=tokens // 2,
        completion_tokens=tokens - tokens // 2,
        total_tokens=tokens,
        cost_usd=cost,
        model=model,
    )


def json_response(obj: Any, **kw: Any) -> ProviderResponse:
    return text_response(json.dumps(obj), **kw)


def tool_call_response(name: str, arguments: dict | str, *, call_id: str = "call_1", **kw: Any) -> ProviderResponse:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    parsed: dict | None
    err: str | None = None
    try:
        loaded = json.loads(raw)
        parsed = loaded if isinstance(loaded, dict) else None
        if parsed is None:
            err = "arguments must be a JSON object"
    except json.JSONDecodeError as exc:
        parsed, err = None, f"arguments are not valid JSON: {exc.msg}"
    resp = text_response(None, **kw)  # type: ignore[arg-type]
    resp.tool_calls = [ToolCallProposal(id=call_id, name=name, raw_arguments=raw, arguments=parsed, parse_error=err)]
    return resp


class FakeProvider:
    """Responses are scripted per model (FIFO). A `default` is used when a model's queue is empty."""

    def __init__(self, script: dict[str, list[Scripted]] | None = None, default: Scripted | None = None):
        self._queues: dict[str, deque[Scripted]] = {m: deque(items) for m, items in (script or {}).items()}
        self._default = default
        self._lock = threading.Lock()
        self.calls: list[LLMRequest] = []

    def enqueue(self, model: str, *items: Scripted) -> None:
        with self._lock:
            self._queues.setdefault(model, deque()).extend(items)

    def supports_response_schema(self, model: str) -> bool:
        return True

    def complete(self, request: LLMRequest) -> ProviderResponse:
        with self._lock:
            self.calls.append(request)
            queue = self._queues.get(request.model)
            item: Scripted | None = queue.popleft() if queue else self._default
        if item is None:
            raise RuntimeError(f"FakeProvider: no scripted response for model '{request.model}'")
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, ProviderResponse):
            return item(request)
        return item
