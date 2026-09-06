"""REST API: resource shapes, status codes, validation, filters/pagination, recheck, trace, metrics."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.db.base import make_engine

PROFILE = {
    "name": "Surfer SEO",
    "domain": "https://www.SurferSEO.com/",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io", "Frase.io"],
}


@pytest.fixture
def client(settings, deps):
    app = create_app(settings, deps=deps, engine=make_engine("sqlite://"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def chaos_client(settings, make_deps):
    # 401 is non-retryable → exactly one AI-check branch degrades regardless of branch scheduling order
    app = create_app(settings, deps=make_deps("llm_responses:401"), engine=make_engine("sqlite://"))
    with TestClient(app) as c:
        yield c


def _create(client) -> str:
    r = client.post("/api/v1/profiles", json=PROFILE)
    assert r.status_code == 201, r.text
    return r.json()["profile_uuid"]


# ── profiles ──────────────────────────────────────────────────────────────────
def test_create_profile_normalises_and_returns_201(client):
    r = client.post("/api/v1/profiles", json=PROFILE)
    assert r.status_code == 201
    body = r.json()
    assert set(body) >= {"profile_uuid", "name", "domain", "status", "created_at"}
    assert body["domain"] == "surferseo.com" and body["status"] == "created"
    assert body["competitors"] == ["clearscope.io", "marketmuse.com", "frase.io"]
    assert r.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "bad",
    [
        {**PROFILE, "name": ""},
        {**PROFILE, "domain": "not a domain"},
        {k: v for k, v in PROFILE.items() if k != "industry"},
        {**PROFILE, "competitors": ["ok.com", "???"]},
        {**PROFILE, "unexpected": 1},
    ],
)
def test_create_profile_validation_422(client, bad):
    r = client.post("/api/v1/profiles", json=bad)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_get_profile_404_and_stats_before_any_run(client):
    assert client.get("/api/v1/profiles/does-not-exist").status_code == 404
    uuid = _create(client)
    body = client.get(f"/api/v1/profiles/{uuid}").json()
    assert body["stats"] == {
        "total_runs": 0,
        "last_run_uuid": None,
        "last_run_status": None,
        "last_run_at": None,
        "average_opportunity_score": None,
        "tracked_queries": 0,
        "visible_queries": 0,
    }


# ── run ───────────────────────────────────────────────────────────────────────
def test_run_pipeline_returns_required_fields(client):
    uuid = _create(client)
    r = client.post(f"/api/v1/profiles/{uuid}/run", json={"question": "How visible is Surfer SEO for SEO tools?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed" and body["mode"] == "full"
    assert body["planned_calls_count"] >= 5 and body["records_extracted_count"] >= 2
    assert body["total_tokens"] > 0 and body["llm_calls_count"] >= 4
    assert body["top_insights"] and {"query_text", "headline", "relevance_score", "opportunity_score"} <= set(
        body["top_insights"][0]
    )
    assert body["report"]["json"]["status"] == "completed" and body["report"]["summary"]
    assert body["degradation"] == [] and body["error"] is None and body["duration_ms"] is not None

    stats = client.get(f"/api/v1/profiles/{uuid}").json()["stats"]
    assert (
        stats["total_runs"] == 1
        and stats["last_run_status"] == "completed"
        and stats["last_run_uuid"] == body["run_uuid"]
    )
    assert stats["tracked_queries"] == body["records_extracted_count"] and 0 <= stats["average_opportunity_score"] <= 1

    assert client.post("/api/v1/profiles/nope/run").status_code == 404
    got = client.get(f"/api/v1/runs/{body['run_uuid']}").json()
    assert got["run_uuid"] == body["run_uuid"] and got["status"] == "completed"
    assert client.get("/api/v1/runs/nope").status_code == 404


def test_run_with_chaos_is_partial_and_persists_degradation(chaos_client):
    uuid = _create(chaos_client)
    body = chaos_client.post(f"/api/v1/profiles/{uuid}/run").json()
    assert body["status"] == "partial"
    assert any("llm_responses" in d for d in body["degradation"])
    trace = chaos_client.get(f"/api/v1/runs/{body['run_uuid']}/trace").json()
    names = [n["node_name"] for n in trace["nodes"]]
    assert "retrieval_fallback" in names and "report" in names
    queries = chaos_client.get(f"/api/v1/profiles/{uuid}/queries").json()["items"]
    assert any(q["error_flag"] and "llm_responses_failed" in q["error_detail"] for q in queries)


# ── queries ───────────────────────────────────────────────────────────────────
def test_queries_sorted_filtered_paginated(client):
    uuid = _create(client)
    client.post(f"/api/v1/profiles/{uuid}/run")
    page = client.get(f"/api/v1/profiles/{uuid}/queries").json()
    items = page["items"]
    assert page["total"] == len(items) >= 2 and page["page"] == 1 and page["per_page"] == 20
    scores = [q["opportunity_score"] for q in items]
    assert scores == sorted(scores, reverse=True)
    required = {
        "query_uuid",
        "query_text",
        "estimated_search_volume",
        "competitive_difficulty",
        "opportunity_score",
        "domain_visible",
        "visibility_position",
        "discovered_at",
    }
    assert all(required <= set(q) for q in items)
    assert all(isinstance(q["estimated_search_volume"], int) and 0 <= q["competitive_difficulty"] <= 100 for q in items)

    threshold = scores[len(scores) // 2]
    filtered = client.get(f"/api/v1/profiles/{uuid}/queries", params={"min_score": threshold}).json()
    assert filtered["total"] == sum(1 for s in scores if s >= threshold)

    for status in ("visible", "not_visible", "unknown"):
        got = client.get(f"/api/v1/profiles/{uuid}/queries", params={"status": status}).json()
        assert all(q["visibility_status"] == status for q in got["items"])
        assert got["total"] == sum(1 for q in items if q["visibility_status"] == status)
    assert client.get(f"/api/v1/profiles/{uuid}/queries", params={"status": "bogus"}).status_code == 422

    p1 = client.get(f"/api/v1/profiles/{uuid}/queries", params={"page": 1, "per_page": 1}).json()
    p2 = client.get(f"/api/v1/profiles/{uuid}/queries", params={"page": 2, "per_page": 1}).json()
    assert (
        len(p1["items"]) == 1 and len(p2["items"]) == 1 and p1["items"][0]["query_uuid"] != p2["items"][0]["query_uuid"]
    )
    assert p1["total"] == page["total"]


def test_rerun_upserts_queries_instead_of_duplicating(client):
    uuid = _create(client)
    client.post(f"/api/v1/profiles/{uuid}/run")
    first = client.get(f"/api/v1/profiles/{uuid}/queries").json()
    client.post(f"/api/v1/profiles/{uuid}/run")
    second = client.get(f"/api/v1/profiles/{uuid}/queries").json()
    assert second["total"] == first["total"]
    assert {q["query_uuid"] for q in first["items"]} == {q["query_uuid"] for q in second["items"]}
    assert client.get(f"/api/v1/profiles/{uuid}").json()["stats"]["total_runs"] == 2
    assert len(client.get(f"/api/v1/profiles/{uuid}/runs").json()) == 2


# ── recommendations ───────────────────────────────────────────────────────────
def test_recommendations_shape_and_linkage(client):
    uuid = _create(client)
    client.post(f"/api/v1/profiles/{uuid}/run")
    recs = client.get(f"/api/v1/profiles/{uuid}/recommendations").json()
    assert recs
    required = {
        "recommendation_uuid",
        "target_query_uuid",
        "content_type",
        "title",
        "rationale",
        "target_keywords",
        "priority",
    }
    assert all(required <= set(r) for r in recs)
    assert all(r["priority"] in {"high", "medium", "low"} for r in recs)
    query_ids = {q["query_uuid"] for q in client.get(f"/api/v1/profiles/{uuid}/queries").json()["items"]}
    assert all(r["target_query_uuid"] in query_ids for r in recs)


# ── recheck ───────────────────────────────────────────────────────────────────
def test_recheck_single_query_updates_record(client):
    uuid = _create(client)
    client.post(f"/api/v1/profiles/{uuid}/run")
    q = client.get(f"/api/v1/profiles/{uuid}/queries").json()["items"][0]

    r = client.post(f"/api/v1/queries/{q['query_uuid']}/recheck")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run"]["mode"] == "recheck" and body["run"]["planned_calls_count"] == 3
    assert body["run"]["records_extracted_count"] == 1
    assert body["query"]["query_uuid"] == q["query_uuid"] and body["query"]["query_text"] == q["query_text"]
    assert body["query"]["last_checked_at"] >= q["last_checked_at"]
    assert body["query"]["last_run_uuid"] == body["run"]["run_uuid"]
    assert all(rec["target_query_uuid"] == q["query_uuid"] for rec in body["recommendations"])

    # profile-level views still show the full picture (latest full run + rechecks since)
    total_after = client.get(f"/api/v1/profiles/{uuid}/queries").json()["total"]
    assert total_after == client.get(f"/api/v1/profiles/{uuid}").json()["stats"]["tracked_queries"]
    assert client.get(f"/api/v1/profiles/{uuid}").json()["stats"]["total_runs"] == 2
    assert client.post("/api/v1/queries/nope/recheck").status_code == 404
    assert client.get(f"/api/v1/queries/{q['query_uuid']}").status_code == 200


# ── observability endpoints ───────────────────────────────────────────────────
def test_trace_and_metrics_endpoints(client):
    uuid = _create(client)
    run_uuid = client.post(f"/api/v1/profiles/{uuid}/run").json()["run_uuid"]
    trace = client.get(f"/api/v1/runs/{run_uuid}/trace").json()
    names = [n["node_name"] for n in trace["nodes"]]
    for node in (
        "query_planner",
        "plan_validator",
        "dispatch_retrieval",
        "retrieval_agent",
        "tool_arg_validator",
        "execute_tool",
        "normalize",
        "normalize_validator",
        "analyze",
        "report",
    ):
        assert node in names
    assert names.index("query_planner") < names.index("normalize") < names.index("report")
    assert all(
        {"duration_ms", "status", "retry_count", "api_calls", "input_summary", "output_summary"} <= set(n)
        for n in trace["nodes"]
    )
    assert client.get("/api/v1/runs/nope/trace").status_code == 404

    m = client.get("/api/v1/metrics").json()
    assert m["runs"] == {"completed": 1}
    assert m["nodes"]["execute_tool"]["count"] >= 5 and m["api_calls"]["serp_organic"]["ok"] >= 1
    assert m["llm_usage"]["total_tokens"] > 0 and isinstance(m["breakers"], list)
    assert client.get("/api/v1/health").json() == {"status": "ok"}
