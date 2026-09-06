"""Error taxonomy shared by the DataForSEO client and the AI gateway.

The distinction that matters for retries:
  * RetryableError    – transient: network/timeout, HTTP 429, HTTP 5xx, provider overload.
  * NonRetryableError – permanent for this request: HTTP 400/401/403/404, malformed payloads,
                        open circuit breaker. Retrying would only burn quota.
"""

from __future__ import annotations


class AppError(Exception):
    kind: str = "app_error"

    def __init__(self, message: str, *, http_status: int | None = None, api_status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.api_status_code = api_status_code

    @property
    def retryable(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "message": self.message,
            "http_status": self.http_status,
            "api_status_code": self.api_status_code,
            "retryable": self.retryable,
        }


class RetryableError(AppError):
    kind = "retryable"

    def __init__(self, message: str, *, kind: str | None = None, **kw: int | None):
        super().__init__(message, **kw)
        if kind:
            self.kind = kind

    @property
    def retryable(self) -> bool:
        return True


class NonRetryableError(AppError):
    kind = "non_retryable"

    def __init__(self, message: str, *, kind: str | None = None, **kw: int | None):
        super().__init__(message, **kw)
        if kind:
            self.kind = kind


class CircuitOpenError(NonRetryableError):
    kind = "breaker_open"


def classify_http_status(status: int) -> bool:
    """Return True when an HTTP status should be retried."""
    return status == 429 or status == 408 or 500 <= status <= 599


def raise_for_http_status(status: int, body_excerpt: str = "") -> None:
    if status < 400:
        return
    msg = f"HTTP {status}: {body_excerpt[:200]}"
    if classify_http_status(status):
        raise RetryableError(msg, kind="http", http_status=status)
    raise NonRetryableError(msg, kind="http", http_status=status)
