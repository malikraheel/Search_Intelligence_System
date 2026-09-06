"""Thread-safe circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED)."""

from __future__ import annotations

import threading
import time
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, *, failure_threshold: int = 5, reset_timeout_s: float = 30.0, clock=time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if self._state is BreakerState.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self.reset_timeout_s:
                self._state = BreakerState.HALF_OPEN
                self._probe_in_flight = False

    def allow(self) -> bool:
        """Whether a call may proceed. In HALF_OPEN exactly one probe call is allowed."""
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.CLOSED:
                return True
            if self._state is BreakerState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = BreakerState.CLOSED
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
                self._probe_in_flight = False

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._maybe_half_open()
            return {"name": self.name, "state": self._state.value, "failures": self._failures}


class BreakerRegistry:
    """One breaker per dependency name (tool endpoint, LLM model, ...)."""

    def __init__(self, *, failure_threshold: int = 5, reset_timeout_s: float = 30.0):
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name, failure_threshold=self._failure_threshold, reset_timeout_s=self._reset_timeout_s
                )
            return self._breakers[name]

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [b.snapshot() for b in self._breakers.values()]
