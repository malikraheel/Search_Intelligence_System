"""Shared fixtures. Everything runs offline: mock/chaos DataForSEO, fake LLM provider, zero backoff."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ["APP_ENV"] = "test"
os.environ["DATAFORSEO_MODE"] = "mock"
os.environ["DATAFORSEO_RETRY_BASE_DELAY_S"] = "0"
os.environ["DATAFORSEO_RETRY_MAX_DELAY_S"] = "0"
os.environ["GATEWAY_RETRY_BASE_DELAY_S"] = "0"
os.environ["GATEWAY_CACHE_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LOG_FORMAT"] = "json"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ.pop("OPENAI_API_KEY", None)

from app.config import Settings, get_settings  # noqa: E402
from app.dataforseo.client import CallContext, ChaosTransport, DataForSEOClient, MockTransport  # noqa: E402
from app.domain.models import ProfileCtx  # noqa: E402
from app.gateway import Gateway  # noqa: E402
from app.gateway.providers.fake import FakeProvider  # noqa: E402
from app.graph.builder import build_graph, initial_state, invoke_graph  # noqa: E402
from app.graph.deps import Deps  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.observability.metrics import MetricsRegistry  # noqa: E402
from app.resilience import BreakerRegistry, RetryPolicy  # noqa: E402
from tests.fakes import make_fake_gateway, smart_llm  # noqa: E402

configure_logging("WARNING", "json")


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return Settings(_env_file=None)


@pytest.fixture
def metrics() -> MetricsRegistry:
    return MetricsRegistry()


@pytest.fixture
def profile() -> ProfileCtx:
    return ProfileCtx(
        profile_uuid="11111111-1111-1111-1111-111111111111",
        name="Surfer SEO",
        domain="surferseo.com",
        industry="SEO Software",
        description="AI-powered SEO content optimization tool",
        competitors=["clearscope.io", "marketmuse.com", "frase.io"],
    )


@pytest.fixture
def call_ctx(profile: ProfileCtx) -> CallContext:
    return CallContext(profile_domain=profile.domain, competitors=profile.competitors)


def make_client(transport, metrics: MetricsRegistry, *, max_attempts: int = 4, threshold: int = 5) -> DataForSEOClient:
    return DataForSEOClient(
        transport,
        policy=RetryPolicy(max_attempts=max_attempts, base_delay_s=0, max_delay_s=0),
        breakers=BreakerRegistry(failure_threshold=threshold, reset_timeout_s=30),
        metrics=metrics,
    )


@pytest.fixture
def mock_client(metrics: MetricsRegistry) -> DataForSEOClient:
    return make_client(MockTransport(), metrics)


@pytest.fixture
def chaos_client_factory(metrics: MetricsRegistry):
    def _make(script: str, **kw) -> DataForSEOClient:
        return make_client(ChaosTransport(MockTransport(), script), metrics, **kw)

    return _make


# ── graph fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture
def fake_llm() -> FakeProvider:
    return FakeProvider(default=smart_llm)


@pytest.fixture
def gateway(metrics: MetricsRegistry, fake_llm: FakeProvider) -> Gateway:
    gw, _ = make_fake_gateway(metrics, fake_llm)
    return gw


@pytest.fixture
def make_deps(settings: Settings, gateway: Gateway, metrics: MetricsRegistry):
    def _make(chaos: str | None = None) -> Deps:
        transport = ChaosTransport(MockTransport(), chaos) if chaos else MockTransport()
        client = make_client(transport, metrics, max_attempts=settings.dataforseo_max_attempts)
        return Deps(settings=settings, gateway=gateway, dfs_client=client, metrics=metrics)

    return _make


@pytest.fixture
def deps(make_deps) -> Deps:
    return make_deps()


@pytest.fixture
def graph(deps: Deps):
    return build_graph(deps)


@pytest.fixture
def run_full(make_deps, profile: ProfileCtx):
    def _run(chaos: str | None = None, max_concurrency: int = 4):
        g = build_graph(make_deps(chaos))
        return invoke_graph(g, initial_state(profile), max_concurrency=max_concurrency)

    return _run


@pytest.fixture
def run_recheck(make_deps, profile: ProfileCtx):
    def _run(query_text: str, chaos: str | None = None):
        g = build_graph(make_deps(chaos))
        return invoke_graph(g, initial_state(profile, mode="recheck", recheck_query=query_text), max_concurrency=1)

    return _run
