"""Retry classification, backoff, and circuit breaker behaviour."""

from __future__ import annotations

import pytest

from app.resilience import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    NonRetryableError,
    RetryableError,
    RetryPolicy,
    call_with_retry,
    classify_http_status,
)
from app.resilience.errors import raise_for_http_status

FAST = RetryPolicy(max_attempts=4, base_delay_s=0, max_delay_s=0)


@pytest.mark.parametrize(
    "status,retryable",
    [
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (408, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_http_status_classification(status, retryable):
    assert classify_http_status(status) is retryable
    with pytest.raises(RetryableError if retryable else NonRetryableError):
        raise_for_http_status(status, "body")


def test_retries_until_success_and_reports_retry_count():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("429", http_status=429)
        return "ok"

    sleeps: list[float] = []
    out = call_with_retry(flaky, policy=FAST, on_retry=lambda a, e, s: sleeps.append(s))
    assert out.value == "ok" and out.attempts == 3 and out.retry_count == 2
    assert len(sleeps) == 2 and all(s == 0 for s in sleeps)


def test_non_retryable_is_raised_immediately():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise NonRetryableError("401", http_status=401)

    with pytest.raises(NonRetryableError):
        call_with_retry(bad, policy=FAST)
    assert calls["n"] == 1


def test_exhausted_retries_reraises_last_error():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise RetryableError("timeout", kind="timeout")

    with pytest.raises(RetryableError):
        call_with_retry(always, policy=FAST)
    assert calls["n"] == FAST.max_attempts


def test_backoff_is_exponential_with_jitter_bounds():
    from tenacity import wait_random_exponential

    wait = wait_random_exponential(multiplier=0.5, max=8)

    class _S:
        def __init__(self, n):
            self.attempt_number = n

    for n in range(1, 8):
        s = wait(_S(n))
        assert 0 <= s <= min(8, 0.5 * 2**n)


def test_breaker_opens_after_threshold_and_half_opens_after_timeout():
    clock = {"t": 0.0}
    b = CircuitBreaker("dep", failure_threshold=3, reset_timeout_s=10, clock=lambda: clock["t"])
    assert b.state is BreakerState.CLOSED
    for _ in range(3):
        b.record_failure()
    assert b.state is BreakerState.OPEN and b.allow() is False
    clock["t"] = 10.5
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow() is True  # single probe
    assert b.allow() is False  # second concurrent probe blocked
    b.record_success()
    assert b.state is BreakerState.CLOSED


def test_breaker_failure_in_half_open_reopens():
    clock = {"t": 0.0}
    b = CircuitBreaker("dep", failure_threshold=1, reset_timeout_s=5, clock=lambda: clock["t"])
    b.record_failure()
    clock["t"] = 5
    assert b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN


def test_call_with_retry_respects_open_breaker():
    b = CircuitBreaker("dep", failure_threshold=2, reset_timeout_s=60)

    def always():
        raise RetryableError("500", http_status=500)

    with pytest.raises(RetryableError):
        call_with_retry(always, policy=RetryPolicy(max_attempts=2, base_delay_s=0, max_delay_s=0), breaker=b)
    assert b.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        call_with_retry(lambda: "never", policy=FAST, breaker=b)
