"""AI gateway: routing, retries, fallback chain, breaker, cache, structured output, accounting."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.gateway import AllProvidersFailed, Gateway, GatewayConfig, Role, RouteConfig, StructuredOutputError
from app.gateway.cache import ResponseCache
from app.gateway.config import provider_of
from app.gateway.errors import ProviderNonRetryable, ProviderRetryable
from app.gateway.providers.fake import FakeProvider, json_response, text_response, tool_call_response
from app.observability.metrics import MetricsRegistry
from app.resilience import BreakerRegistry

PRIMARY = "fake/primary"
FALLBACK = "fake/fallback"
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def make_gateway(provider: FakeProvider, *, cache: bool = False, threshold: int = 5, retries: int = 2) -> Gateway:
    routes = {r: RouteConfig(model=PRIMARY, fallbacks=[FALLBACK], max_retries=retries) for r in Role}
    cfg = GatewayConfig(routes=routes, retry_base_delay_s=0, cache_enabled=cache, breaker_failure_threshold=threshold)
    return Gateway(
        cfg,
        {"fake": provider},
        metrics=MetricsRegistry(),
        cache=ResponseCache(600) if cache else None,
        breakers=BreakerRegistry(failure_threshold=threshold, reset_timeout_s=60),
    )


def test_provider_prefix_parsing():
    assert provider_of("openai/gpt-4o-mini") == "openai"
    assert provider_of("gpt-4o") == "openai"
    assert provider_of("anthropic/claude-3-5-sonnet") == "anthropic"


def test_routes_to_primary_and_accounts_usage():
    fake = FakeProvider({PRIMARY: [text_response("hello", tokens=120, cost=0.002)]})
    gw = make_gateway(fake)
    resp = gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r1")
    assert resp.text == "hello" and resp.model == PRIMARY and not resp.fallback_used and resp.attempts == 1
    assert resp.usage.total_tokens == 120 and resp.usage.cost_usd == 0.002 and not resp.usage.cost_unknown
    assert gw.ledger.totals("r1")["total_tokens"] == 120
    assert fake.calls[0].model == PRIMARY and fake.calls[0].temperature == 0


def test_retryable_errors_are_retried_on_same_model():
    fake = FakeProvider({PRIMARY: [ProviderRetryable("429"), ProviderRetryable("503"), text_response("ok")]})
    resp = make_gateway(fake).complete(role=Role.ANALYST, messages=MSGS, run_id="r")
    assert resp.text == "ok" and resp.attempts == 3 and not resp.fallback_used


def test_non_retryable_error_falls_back_immediately():
    fake = FakeProvider({PRIMARY: [ProviderNonRetryable("401 auth")], FALLBACK: [text_response("from fallback")]})
    resp = make_gateway(fake).complete(role=Role.PLANNER, messages=MSGS, run_id="r")
    assert resp.text == "from fallback" and resp.fallback_used and resp.model == FALLBACK
    assert [c.model for c in fake.calls] == [PRIMARY, FALLBACK]


def test_exhausted_retries_fall_back_to_next_model():
    fake = FakeProvider({PRIMARY: [ProviderRetryable("500")] * 3, FALLBACK: [text_response("fb")]})
    resp = make_gateway(fake, retries=2).complete(role=Role.PLANNER, messages=MSGS, run_id="r")
    assert resp.model == FALLBACK and len(fake.calls) == 4


def test_all_models_failing_raises_all_providers_failed():
    fake = FakeProvider({PRIMARY: [ProviderNonRetryable("bad")], FALLBACK: [ProviderNonRetryable("worse")]})
    with pytest.raises(AllProvidersFailed) as exc:
        make_gateway(fake).complete(role=Role.REPORTER, messages=MSGS, run_id="r")
    assert set(exc.value.errors_by_model) == {PRIMARY, FALLBACK}


def test_open_breaker_skips_model():
    fake = FakeProvider(
        {PRIMARY: [ProviderRetryable("500")] * 10, FALLBACK: [text_response("fb1"), text_response("fb2")]}
    )
    gw = make_gateway(fake, threshold=3, retries=2)
    first = gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r")  # 3 failures → breaker opens → fallback
    assert first.model == FALLBACK
    calls_before = len(fake.calls)
    second = gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r")
    assert second.model == FALLBACK
    assert len(fake.calls) == calls_before + 1  # primary skipped entirely (circuit open)
    assert gw.breakers.get(PRIMARY).state.value == "open"


def test_cache_hit_avoids_provider_and_zero_cost():
    fake = FakeProvider({PRIMARY: [text_response("cached-me", tokens=50, cost=0.01)]})
    gw = make_gateway(fake, cache=True)
    a = gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r1")
    b = gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r2")
    assert not a.cached and b.cached and b.text == "cached-me"
    assert len(fake.calls) == 1
    assert gw.ledger.totals("r2")["cost_usd"] == 0.0 and gw.ledger.totals("r2")["cached_calls"] == 1


def test_non_deterministic_requests_are_not_cached():
    fake = FakeProvider(default=text_response("x"))
    gw = make_gateway(fake, cache=True)
    gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r", temperature=0.7)
    gw.complete(role=Role.PLANNER, messages=MSGS, run_id="r", temperature=0.7)
    assert len(fake.calls) == 2


def test_tool_calls_are_surfaced_including_malformed_arguments():
    fake = FakeProvider({PRIMARY: [tool_call_response("serp_organic", '{"keyword": "x", ')]})
    resp = make_gateway(fake).complete(role=Role.RETRIEVAL, messages=MSGS, run_id="r", tools=[{"type": "function"}])
    assert resp.tool_calls[0].name == "serp_organic"
    assert resp.tool_calls[0].arguments is None and "not valid JSON" in resp.tool_calls[0].parse_error
    assert fake.calls[0].tools and fake.calls[0].tool_choice == None  # noqa: E711 – provider defaults to "auto"


class Plan(BaseModel):
    items: list[str] = Field(min_length=1)
    score: float = Field(ge=0, le=1)


def test_structured_output_is_validated_and_parsed():
    fake = FakeProvider({PRIMARY: [json_response({"items": ["a"], "score": 0.5})]})
    resp = make_gateway(fake).complete(role=Role.PLANNER, messages=MSGS, run_id="r", response_model=Plan)
    assert isinstance(resp.parsed, Plan) and resp.parsed.items == ["a"]
    assert fake.calls[0].response_format is Plan


def test_structured_output_repair_round_trip():
    fake = FakeProvider(
        {
            PRIMARY: [
                text_response('```json\n{"items": [], "score": 2}\n```'),
                json_response({"items": ["b"], "score": 1}),
            ]
        }
    )
    resp = make_gateway(fake).complete(role=Role.PLANNER, messages=MSGS, run_id="r", response_model=Plan)
    assert resp.parsed.items == ["b"] and resp.attempts == 2
    repair_prompt = fake.calls[1].messages[-1]["content"]
    assert "invalid" in repair_prompt and "score" in repair_prompt


def test_structured_output_failure_after_repair_falls_back_then_raises():
    fake = FakeProvider(
        {
            PRIMARY: [text_response("not json"), text_response("still not json")],
            FALLBACK: [json_response({"items": ["ok"], "score": 0})],
        }
    )
    resp = make_gateway(fake).complete(role=Role.PLANNER, messages=MSGS, run_id="r", response_model=Plan)
    assert resp.fallback_used and resp.parsed.items == ["ok"]

    fake2 = FakeProvider(default=text_response("garbage"))
    with pytest.raises(AllProvidersFailed) as exc:
        make_gateway(fake2).complete(role=Role.PLANNER, messages=MSGS, run_id="r", response_model=Plan)
    assert StructuredOutputError.__name__ in next(iter(exc.value.errors_by_model.values()))


def test_schema_injected_when_provider_lacks_native_support():
    class NoSchema(FakeProvider):
        def supports_response_schema(self, model: str) -> bool:
            return False

    fake = NoSchema({PRIMARY: [json_response({"items": ["a"], "score": 0.1})]})
    make_gateway(fake).complete(role=Role.PLANNER, messages=MSGS, run_id="r", response_model=Plan)
    req = fake.calls[0]
    assert req.response_format == {"type": "json_object"}
    assert "JSON schema" in req.messages[0]["content"] and '"items"' in req.messages[0]["content"]


def test_config_from_settings_drops_fallbacks_without_keys(settings):
    settings = settings.model_copy(
        update={
            "openai_api_key": "sk",
            "anthropic_api_key": None,
            "gateway_planner_fallbacks": "anthropic/claude-x,openai/gpt-4o",
        }
    )
    cfg = GatewayConfig.from_settings(settings)
    assert cfg.routes[Role.PLANNER].chain == ["openai/gpt-4o-mini", "openai/gpt-4o"]
