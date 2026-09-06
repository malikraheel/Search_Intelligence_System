"""Retry with exponential backoff + full jitter, integrated with the circuit breaker.

Only `RetryableError` triggers a retry. `NonRetryableError` (incl. CircuitOpenError) is raised
immediately. The caller receives the number of retries performed for logging/metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.resilience.breaker import CircuitBreaker
from app.resilience.errors import CircuitOpenError, RetryableError


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4  # 1 initial + (max_attempts-1) retries
    base_delay_s: float = 0.5  # sleep ∈ [0, min(max_delay, base * 2**n)] — full jitter
    max_delay_s: float = 8.0


@dataclass
class RetryOutcome[T]:
    value: T
    attempts: int

    @property
    def retry_count(self) -> int:
        return max(self.attempts - 1, 0)


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> RetryOutcome[T]:
    """Run `fn` with backoff. Raises the last error after exhausting attempts."""
    attempts = 0

    def _attempt() -> T:
        nonlocal attempts
        if breaker is not None and not breaker.allow():
            raise CircuitOpenError(f"circuit '{breaker.name}' is open", kind="breaker_open")
        attempts += 1
        try:
            result = fn()
        except RetryableError:
            if breaker is not None:
                breaker.record_failure()
            raise
        if breaker is not None:
            breaker.record_success()
        return result

    def _before_sleep(state: RetryCallState) -> None:
        if on_retry and state.outcome is not None and state.next_action is not None:
            on_retry(state.attempt_number, state.outcome.exception(), float(state.next_action.sleep))

    retrying = Retrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_random_exponential(multiplier=policy.base_delay_s, max=policy.max_delay_s),
        retry=retry_if_exception_type(RetryableError),
        before_sleep=_before_sleep,
        reraise=True,
    )
    value = retrying(_attempt)
    return RetryOutcome(value=value, attempts=attempts)
