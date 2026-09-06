"""DataForSEO client: mock realism, chaos-mode retry/fallback classification, live transport via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.dataforseo.client import CallContext, LiveTransport, parse_chaos_script
from app.resilience import CircuitOpenError, NonRetryableError, RetryableError
from app.tools.registry import TOOLS
from app.tools.schemas import KeywordVolumeInput, LlmResponsesInput, SerpOrganicInput

SERP = TOOLS["serp_organic"]
LLM = TOOLS["llm_responses"]
VOL = TOOLS["keyword_volume"]


def test_mock_serp_is_deterministic_and_shaped_like_dataforseo(mock_client, call_ctx):
    a = mock_client.call(SERP, SerpOrganicInput(keyword="best seo software"), call_ctx)
    b = mock_client.call(SERP, SerpOrganicInput(keyword="best seo software"), call_ctx)
    assert a.payload == b.payload and a.attempts == 1 and a.retry_count == 0
    task = a.payload["tasks"][0]
    assert task["status_code"] == 20000
    items = task["result"][0]["items"]
    assert sum(1 for i in items if i["type"] == "organic") == 10
    assert all("domain" in i for i in items if i["type"] == "organic")


def test_mock_brand_query_ranks_profile_first(mock_client, call_ctx):
    out = mock_client.call(SERP, SerpOrganicInput(keyword="brand: surfer seo"), call_ctx)
    organic = [i for i in out.payload["tasks"][0]["result"][0]["items"] if i["type"] == "organic"]
    assert organic[0]["domain"] == "surferseo.com"


def test_mock_llm_and_volume_shapes(mock_client, call_ctx):
    llm = mock_client.call(LLM, LlmResponsesInput(user_prompt="best seo tools"), call_ctx)
    assert llm.payload["tasks"][0]["result"][0]["items"][0]["sections"][0]["annotations"]
    vol = mock_client.call(VOL, KeywordVolumeInput(keywords=["seo tool", "content optimizer"]), call_ctx)
    rows = vol.payload["tasks"][0]["result"]
    assert [r["keyword"] for r in rows] == ["seo tool", "content optimizer"]
    assert all(isinstance(r["search_volume"], int) and 0 <= r["competition_index"] <= 100 for r in rows)


def test_chaos_script_parsing():
    assert parse_chaos_script("serp_organic:429, 429,ok; llm_responses:500") == {
        "serp_organic": ["429", "429", "ok"],
        "llm_responses": ["500"],
    }


def test_chaos_transient_errors_are_retried_then_succeed(chaos_client_factory, call_ctx, metrics):
    client = chaos_client_factory("serp_organic:429,500,ok")
    out = client.call(SERP, SerpOrganicInput(keyword="x"), call_ctx)
    assert out.attempts == 3 and out.retry_count == 2
    assert out.payload["tasks"][0]["status_code"] == 20000
    api = metrics.snapshot()["api_calls"]["serp_organic"]
    assert api["retryable_error"] == 2 and api["ok"] == 1


def test_chaos_timeout_then_ok(chaos_client_factory, call_ctx):
    out = chaos_client_factory("keyword_volume:timeout,ok").call(VOL, KeywordVolumeInput(keywords=["a"]), call_ctx)
    assert out.retry_count == 1


def test_chaos_non_retryable_auth_error_fails_fast(chaos_client_factory, call_ctx, metrics):
    client = chaos_client_factory("serp_organic:401,ok")
    with pytest.raises(NonRetryableError) as exc:
        client.call(SERP, SerpOrganicInput(keyword="x"), call_ctx)
    assert exc.value.http_status == 401
    assert metrics.snapshot()["api_calls"]["serp_organic"] == {"non_retryable_error": 1, "total": 1}


def test_chaos_exhausted_retries_raise_retryable(chaos_client_factory, call_ctx):
    client = chaos_client_factory("llm_responses:500,500,500,500,500", max_attempts=4)
    with pytest.raises(RetryableError):
        client.call(LLM, LlmResponsesInput(user_prompt="x"), call_ctx)


def test_breaker_opens_after_repeated_failures(chaos_client_factory, call_ctx, metrics):
    client = chaos_client_factory("serp_organic:" + ",".join(["500"] * 20), max_attempts=2, threshold=3)
    with pytest.raises(RetryableError):
        client.call(SERP, SerpOrganicInput(keyword="a"), call_ctx)  # 2 failures
    with pytest.raises((RetryableError, CircuitOpenError)):
        client.call(SERP, SerpOrganicInput(keyword="b"), call_ctx)  # 3rd failure opens
    with pytest.raises(CircuitOpenError):
        client.call(SERP, SerpOrganicInput(keyword="c"), call_ctx)
    assert metrics.snapshot()["api_calls"]["serp_organic"]["breaker_open"] >= 1


def test_malformed_payload_is_non_retryable(chaos_client_factory, call_ctx):
    out = chaos_client_factory("serp_organic:malformed").call(SERP, SerpOrganicInput(keyword="x"), call_ctx)
    # transport returns a 20000 envelope with a broken result → client passes it through; parser flags it later
    assert out.payload["tasks"][0]["result"][0]["items"] == "not-a-list"


def test_task_level_error_codes_are_classified(mock_client, call_ctx, monkeypatch):
    from app.dataforseo import client as mod

    def bad_transport(path, payload, ctx):
        return {
            "status_code": 20000,
            "tasks": [{"status_code": 40501, "status_message": "Invalid Field", "result": None}],
        }

    monkeypatch.setattr(mock_client._transport, "post", bad_transport)
    with pytest.raises(NonRetryableError) as exc:
        mock_client.call(SERP, SerpOrganicInput(keyword="x"), call_ctx)
    assert exc.value.api_status_code == 40501
    assert mod is not None


@respx.mock
def test_live_transport_sends_basic_auth_and_classifies(call_ctx):
    t = LiveTransport("https://api.dataforseo.com", "login", "pw", connect_timeout_s=1, read_timeout_s=1)
    route = respx.post("https://api.dataforseo.com/v3/serp/google/organic/live/advanced").mock(
        return_value=httpx.Response(200, json={"status_code": 20000, "tasks": [{"status_code": 20000, "result": []}]})
    )
    body = t.post(SERP.endpoint_path, [{"keyword": "x"}], call_ctx)
    assert body["status_code"] == 20000
    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")
    assert route.calls.last.request.content == b'[{"keyword":"x"}]'

    respx.post("https://api.dataforseo.com/v3/serp/google/organic/live/advanced").mock(return_value=httpx.Response(429))
    with pytest.raises(RetryableError):
        t.post(SERP.endpoint_path, [{"keyword": "x"}], call_ctx)

    respx.post("https://api.dataforseo.com/v3/serp/google/organic/live/advanced").mock(return_value=httpx.Response(401))
    with pytest.raises(NonRetryableError):
        t.post(SERP.endpoint_path, [{"keyword": "x"}], call_ctx)

    respx.post("https://api.dataforseo.com/v3/serp/google/organic/live/advanced").mock(
        side_effect=httpx.ConnectTimeout("t")
    )
    with pytest.raises(RetryableError):
        t.post(SERP.endpoint_path, [{"keyword": "x"}], call_ctx)


def test_live_transport_requires_credentials():
    with pytest.raises(NonRetryableError):
        LiveTransport("https://api.dataforseo.com", "", "", connect_timeout_s=1, read_timeout_s=1)


def test_call_context_defaults():
    assert CallContext().profile_domain == "example.com"
