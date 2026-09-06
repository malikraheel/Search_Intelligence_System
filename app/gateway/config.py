from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.gateway.types import Role

# Providers that don't need an API key to be usable.
_KEYLESS_PROVIDERS = {"fake", "ollama", "ollama_chat"}


def provider_of(model: str) -> str:
    """'openai/gpt-4o-mini' → 'openai'. Models without a prefix are assumed to be OpenAI."""
    return model.split("/", 1)[0] if "/" in model else "openai"


@dataclass(frozen=True)
class RouteConfig:
    model: str
    fallbacks: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int | None = 2048
    timeout_s: float = 30.0
    max_retries: int = 2

    @property
    def chain(self) -> list[str]:
        return [self.model, *self.fallbacks]


@dataclass
class GatewayConfig:
    routes: dict[Role, RouteConfig]
    retry_base_delay_s: float = 0.5
    cache_enabled: bool = True
    cache_ttl_s: int = 600
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0

    @classmethod
    def from_settings(cls, settings: Settings) -> GatewayConfig:
        keys = settings.provider_keys()

        def usable(model: str) -> bool:
            prov = provider_of(model)
            return prov in _KEYLESS_PROVIDERS or bool(keys.get(prov))

        routes: dict[Role, RouteConfig] = {}
        for role in Role:
            primary = settings.gateway_model(role.value)
            # Primary is kept even without a key (fails fast with an auth error → fallback);
            # fallbacks without a configured key are dropped so we never route to a dead end.
            fallbacks = [m for m in settings.gateway_fallbacks(role.value) if usable(m) and m != primary]
            routes[role] = RouteConfig(
                model=primary,
                fallbacks=fallbacks,
                temperature=0.0,
                max_tokens=4096 if role in (Role.ANALYST, Role.REPORTER) else 2048,
                timeout_s=settings.gateway_timeout_s,
                max_retries=settings.gateway_max_retries,
            )
        return cls(
            routes=routes,
            retry_base_delay_s=settings.gateway_retry_base_delay_s,
            cache_enabled=settings.gateway_cache_enabled,
            cache_ttl_s=settings.gateway_cache_ttl_s,
            breaker_failure_threshold=settings.gateway_breaker_failure_threshold,
            breaker_reset_s=settings.gateway_breaker_reset_s,
        )
