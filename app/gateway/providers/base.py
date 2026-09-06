from __future__ import annotations

from typing import Protocol

from app.gateway.types import LLMRequest, ProviderResponse


class LLMProvider(Protocol):
    """Adapter contract. Implementations raise ProviderRetryable / ProviderNonRetryable."""

    def complete(self, request: LLMRequest) -> ProviderResponse: ...

    def supports_response_schema(self, model: str) -> bool: ...
