from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from app.domain.models import LLMUsage


class UsageLedger:
    """Per-run and global token/cost accounting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, list[LLMUsage]] = defaultdict(list)

    def add(self, run_id: str, usage: LLMUsage) -> None:
        with self._lock:
            self._by_run[run_id].append(usage)

    def totals(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            items = list(self._by_run.get(run_id, []))
        return _summarise(items)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            everything = [u for items in self._by_run.values() for u in items]
            runs = len(self._by_run)
        return {"runs": runs, **_summarise(everything)}

    def forget(self, run_id: str) -> None:
        with self._lock:
            self._by_run.pop(run_id, None)


def _summarise(items: list[LLMUsage]) -> dict[str, Any]:
    by_role: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost_usd": 0.0})
    for u in items:
        by_role[u.role]["calls"] += 1
        by_role[u.role]["tokens"] += u.total_tokens
        by_role[u.role]["cost_usd"] += u.cost_usd
    return {
        "llm_calls": len(items),
        "prompt_tokens": sum(u.prompt_tokens for u in items),
        "completion_tokens": sum(u.completion_tokens for u in items),
        "total_tokens": sum(u.total_tokens for u in items),
        "cost_usd": round(sum(u.cost_usd for u in items), 6),
        "cost_unknown": any(u.cost_unknown for u in items),
        "cached_calls": sum(1 for u in items if u.cached),
        "by_role": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_role.items()},
    }
