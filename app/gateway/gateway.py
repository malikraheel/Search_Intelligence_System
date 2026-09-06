"""The Gateway: routing → cache → breaker → provider chain with retries → validation → accounting."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.domain.models import LLMUsage
from app.gateway.cache import ResponseCache, request_key
from app.gateway.config import GatewayConfig, provider_of
from app.gateway.errors import AllProvidersFailed, ProviderNonRetryable, StructuredOutputError
from app.gateway.providers.base import LLMProvider
from app.gateway.types import LLMRequest, LLMResponse, ProviderResponse, Role
from app.gateway.usage import UsageLedger
from app.observability.logging import get_logger
from app.observability.metrics import MetricsRegistry
from app.resilience import BreakerRegistry, CircuitOpenError, RetryableError, RetryPolicy, call_with_retry

log = get_logger("gateway")


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, LLMProvider],
        *,
        metrics: MetricsRegistry,
        ledger: UsageLedger | None = None,
        cache: ResponseCache | None = None,
        breakers: BreakerRegistry | None = None,
    ):
        self._config = config
        self._providers = providers
        self._metrics = metrics
        self.ledger = ledger or UsageLedger()
        self._cache = (
            cache if cache is not None else (ResponseCache(config.cache_ttl_s) if config.cache_enabled else None)
        )
        self.breakers = breakers or BreakerRegistry(
            failure_threshold=config.breaker_failure_threshold, reset_timeout_s=config.breaker_reset_s
        )

    # ── public API ────────────────────────────────────────────────────────────
    def complete(
        self,
        *,
        role: Role,
        messages: list[dict[str, Any]],
        run_id: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
        use_cache: bool | None = None,
    ) -> LLMResponse:
        route = self._config.routes[role]
        temp = route.temperature if temperature is None else temperature
        cacheable = (use_cache if use_cache is not None else True) and self._cache is not None and temp == 0
        key = request_key(route.model, messages, tools, response_model) if cacheable else None

        if key and self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                cached = hit.model_copy(update={"cached": True})
                cached.usage = hit.usage.model_copy(update={"cached": True, "cost_usd": 0.0})
                self.ledger.add(run_id, cached.usage)
                log.info("llm.call", role=role.value, model=hit.model, cached=True, run_id=run_id)
                return cached

        errors_by_model: dict[str, str] = {}
        for idx, model in enumerate(route.chain):
            breaker = self.breakers.get(model)
            if not breaker.allow():
                errors_by_model[model] = "circuit open"
                log.warning("llm.skip_model", role=role.value, model=model, reason="circuit_open")
                continue
            provider = self._provider_for(model)
            if provider is None:
                errors_by_model[model] = "no provider configured"
                continue

            started = time.perf_counter()
            try:
                result, attempts = self._call_model(
                    provider, model, route, messages, tools, tool_choice, response_model, temp, breaker
                )
            except (RetryableError, ProviderNonRetryable, CircuitOpenError, StructuredOutputError) as exc:
                errors_by_model[model] = f"{type(exc).__name__}: {exc}"
                latency = (time.perf_counter() - started) * 1000
                self._metrics.record_llm_call(role.value, model, latency, ok=False)
                log.warning(
                    "llm.model_failed", role=role.value, model=model, error=str(exc), latency_ms=round(latency, 1)
                )
                continue

            latency = (time.perf_counter() - started) * 1000
            response = self._to_response(result, role, model, attempts, idx > 0, latency, response_model)
            self.ledger.add(run_id, response.usage)
            self._metrics.record_llm_call(role.value, model, latency, ok=True, tokens=response.usage.total_tokens)
            log.info(
                "llm.call",
                role=role.value,
                model=model,
                attempts=attempts,
                fallback_used=idx > 0,
                tokens=response.usage.total_tokens,
                cost_usd=response.usage.cost_usd,
                latency_ms=round(latency, 1),
                tool_calls=[t.name for t in response.tool_calls],
                run_id=run_id,
            )
            if key and self._cache is not None:
                self._cache.set(key, response)
            return response

        raise AllProvidersFailed(role.value, errors_by_model)

    # ── internals ─────────────────────────────────────────────────────────────
    def _provider_for(self, model: str) -> LLMProvider | None:
        return self._providers.get(provider_of(model)) or self._providers.get("default")

    def _call_model(
        self,
        provider: LLMProvider,
        model: str,
        route: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_model: type[BaseModel] | None,
        temperature: float,
        breaker: Any,
    ) -> tuple[ProviderResponse, int]:
        msgs = list(messages)
        response_format: Any = None
        if response_model is not None:
            if provider.supports_response_schema(model):
                response_format = response_model
            else:
                response_format = {"type": "json_object"}
                msgs = _inject_schema(msgs, response_model)

        policy = RetryPolicy(
            max_attempts=route.max_retries + 1, base_delay_s=self._config.retry_base_delay_s, max_delay_s=8
        )
        total_attempts = 0

        def _invoke(messages_: list[dict[str, Any]]) -> ProviderResponse:
            nonlocal total_attempts
            req = LLMRequest(
                model=model,
                messages=messages_,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                temperature=temperature,
                max_tokens=route.max_tokens,
                timeout_s=route.timeout_s,
            )
            outcome = call_with_retry(
                lambda: provider.complete(req),
                policy=policy,
                breaker=breaker,
                on_retry=lambda a, e, s: log.warning(
                    "llm.retry", model=model, attempt=a, sleep_s=round(s, 2), error=str(e)
                ),
            )
            total_attempts += outcome.attempts
            return outcome.value

        result = _invoke(msgs)
        if response_model is None:
            return result, total_attempts

        # Structured output: validate; on failure do ONE repair round-trip with the error text.
        try:
            _validate_structured(result, response_model)
            return result, total_attempts
        except StructuredOutputError as first_err:
            log.warning("llm.structured_repair", model=model, error=str(first_err))
            repair_msgs = msgs + [
                {"role": "assistant", "content": result.text or ""},
                {
                    "role": "user",
                    "content": (
                        f"Your previous JSON was invalid: {first_err}. "
                        "Reply with ONLY corrected JSON matching the schema."
                    ),
                },
            ]
            result = _invoke(repair_msgs)
            _validate_structured(result, response_model)  # raises StructuredOutputError if still bad
            return result, total_attempts

    def _to_response(
        self,
        result: ProviderResponse,
        role: Role,
        model: str,
        attempts: int,
        fallback_used: bool,
        latency_ms: float,
        response_model: type[BaseModel] | None,
    ) -> LLMResponse:
        parsed = response_model.model_validate_json(_strip_fences(result.text or "")) if response_model else None
        usage = LLMUsage(
            role=role.value,
            model=result.model or model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens or (result.prompt_tokens + result.completion_tokens),
            cost_usd=result.cost_usd or 0.0,
            cost_unknown=result.cost_usd is None,
        )
        return LLMResponse(
            text=result.text,
            tool_calls=result.tool_calls,
            parsed=parsed,
            usage=usage,
            model=model,
            attempts=attempts,
            fallback_used=fallback_used,
            latency_ms=round(latency_ms, 1),
        )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _validate_structured(result: ProviderResponse, model_cls: type[BaseModel]) -> None:
    if not result.text:
        raise StructuredOutputError("empty response where JSON was expected")
    try:
        model_cls.model_validate_json(_strip_fences(result.text))
    except ValidationError as exc:
        errs = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
        raise StructuredOutputError(f"schema validation failed: {errs}") from exc


def _inject_schema(messages: list[dict[str, Any]], model_cls: type[BaseModel]) -> list[dict[str, Any]]:
    schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
    note = f"\n\nRespond ONLY with a JSON object matching this JSON schema:\n{schema}"
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0].get('content', '')}{note}"
    else:
        out.insert(0, {"role": "system", "content": note.strip()})
    return out


def build_gateway(
    settings: Settings, metrics: MetricsRegistry, providers: dict[str, LLMProvider] | None = None
) -> Gateway:
    config = GatewayConfig.from_settings(settings)
    if providers is None:
        from app.gateway.providers.litellm_provider import LiteLLMProvider  # lazy: slow import

        lite = LiteLLMProvider(settings.provider_keys())
        providers = {"default": lite}
    return Gateway(config, providers, metrics=metrics)
