"""In-memory metrics registry: per-node latency + success/failure, API and LLM call counts."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class _Stat:
    __slots__ = ("count", "success", "failure", "latency_sum", "latency_max", "latencies")

    def __init__(self) -> None:
        self.count = 0
        self.success = 0
        self.failure = 0
        self.latency_sum = 0.0
        self.latency_max = 0.0
        self.latencies: list[float] = []

    def add(self, latency_ms: float, ok: bool) -> None:
        self.count += 1
        self.success += int(ok)
        self.failure += int(not ok)
        self.latency_sum += latency_ms
        self.latency_max = max(self.latency_max, latency_ms)
        if len(self.latencies) < 1000:
            self.latencies.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        p95 = 0.0
        if self.latencies:
            ordered = sorted(self.latencies)
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        return {
            "count": self.count,
            "success": self.success,
            "failure": self.failure,
            "success_rate": round(self.success / self.count, 3) if self.count else None,
            "latency_ms_avg": round(self.latency_sum / self.count, 1) if self.count else None,
            "latency_ms_p95": round(p95, 1),
            "latency_ms_max": round(self.latency_max, 1),
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, _Stat] = defaultdict(_Stat)
        self._api: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._llm: dict[str, _Stat] = defaultdict(_Stat)
        self._llm_tokens: dict[str, int] = defaultdict(int)
        self._runs: dict[str, int] = defaultdict(int)

    def record_node(self, node: str, latency_ms: float, ok: bool) -> None:
        with self._lock:
            self._nodes[node].add(latency_ms, ok)

    def record_api_call(self, tool: str, outcome: str) -> None:
        with self._lock:
            self._api[tool][outcome] += 1
            self._api[tool]["total"] += 1

    def record_llm_call(self, role: str, model: str, latency_ms: float, ok: bool, tokens: int = 0) -> None:
        with self._lock:
            key = f"{role}:{model}"
            self._llm[key].add(latency_ms, ok)
            self._llm_tokens[key] += tokens

    def record_run(self, status: str) -> None:
        with self._lock:
            self._runs[status] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "runs": dict(self._runs),
                "nodes": {k: v.snapshot() for k, v in self._nodes.items()},
                "api_calls": {k: dict(v) for k, v in self._api.items()},
                "llm_calls": {k: {**v.snapshot(), "tokens": self._llm_tokens[k]} for k, v in self._llm.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._api.clear()
            self._llm.clear()
            self._llm_tokens.clear()
            self._runs.clear()
