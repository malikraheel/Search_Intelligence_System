"""AI Gateway: the single entry point for every LLM call made by the agents.

Responsibilities: per-role model routing, provider fallback chain, retries with backoff,
circuit breaking per model, response caching, structured-output validation, and
token/cost accounting. Agents never import a provider SDK directly.
"""

from app.gateway.config import GatewayConfig, RouteConfig
from app.gateway.errors import AllProvidersFailed, GatewayError, StructuredOutputError
from app.gateway.gateway import Gateway, build_gateway
from app.gateway.types import LLMRequest, LLMResponse, ProviderResponse, Role
from app.gateway.usage import UsageLedger

__all__ = [
    "AllProvidersFailed",
    "Gateway",
    "GatewayConfig",
    "GatewayError",
    "LLMRequest",
    "LLMResponse",
    "ProviderResponse",
    "Role",
    "RouteConfig",
    "StructuredOutputError",
    "UsageLedger",
    "build_gateway",
]
