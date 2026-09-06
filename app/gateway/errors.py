from __future__ import annotations

from app.resilience.errors import AppError, NonRetryableError, RetryableError


class GatewayError(AppError):
    kind = "gateway"


class ProviderRetryable(RetryableError):
    """Rate limit, overload, timeout, connection error – safe to retry."""

    kind = "provider_retryable"


class ProviderNonRetryable(NonRetryableError):
    """Auth, bad request, content policy, context window – retrying won't help."""

    kind = "provider_non_retryable"


class AllProvidersFailed(NonRetryableError):
    kind = "all_providers_failed"

    def __init__(self, role: str, errors_by_model: dict[str, str]):
        self.role = role
        self.errors_by_model = errors_by_model
        detail = "; ".join(f"{m}: {e}" for m, e in errors_by_model.items())
        super().__init__(f"all models failed for role '{role}': {detail}", kind=self.kind)


class StructuredOutputError(NonRetryableError):
    kind = "structured_output"
