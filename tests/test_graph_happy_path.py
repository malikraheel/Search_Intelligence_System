"""Full DAG, mock DataForSEO, fake LLM → completed run with all agents contributing."""

from __future__ import annotations

from app.graph.builder import mermaid

EXPECTED_NODES = {
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
}
FALLBACK_NODES = {"plan_repair", "fallback_plan", "retrieval_fallback", "analyze_fallback", "degraded_report"}


def test_happy_path_completed(run_full):
    out = run_full()
    plan = out["plan"]
    assert out["status"] == "completed"
    assert out["plan_source"] == "llm" and 2 <= len(plan.sub_queries) <= 8

    # 1 SERP per query + 1 AI check per wants_ai_check + 1 batched volume call
    expected_calls = len(plan.sub_queries) + sum(sq.wants_ai_check for sq in plan.sub_queries) + 1
    assert len(out["planned_calls"]) == expected_calls
    assert len(out["raw_results"]) == expected_calls
    assert all(r.status == "ok" for r in out["raw_results"])

    records = out["records"]
    assert len(records) == len(plan.sub_queries) and out["dropped_records"] == 0
    for rec in records:
        assert 0 <= rec.opportunity_score <= 1 and 0 <= rec.competitive_difficulty <= 100
        assert rec.estimated_search_volume is not None
        assert rec.visibility_status in {"visible", "not_visible"}
        assert "serp_organic" in rec.sources and "keyword_volume" in rec.sources
        assert not rec.error_flags

    analysis = out["analysis"]
    assert out["analysis_source"] == "llm" and analysis.insights and analysis.recommendations
    assert all(any(i.query_text == r.query_text for r in records) for i in analysis.insights)

    report = out["report"]
    assert report.report_json["status"] == "completed" and report.report_json["records_extracted"] == len(records)
    assert report.report_json["token_usage"]["total_tokens"] > 0
    assert "Summary" in report.summary or "visibility report" in report.summary
    assert report.report_json["degradation"] == []


def test_happy_path_traces_cover_every_agent_and_no_fallbacks(run_full):
    out = run_full()
    seen = {t.node for t in out["node_traces"]}
    assert EXPECTED_NODES <= seen
    assert not (seen & FALLBACK_NODES)
    branch_traces = [t for t in out["node_traces"] if t.node == "execute_tool"]
    assert len(branch_traces) == len(out["planned_calls"])
    assert all(t.status == "success" and t.duration_ms >= 0 for t in out["node_traces"])
    assert all(t.branch_key for t in branch_traces)


def test_llm_usage_is_accounted_per_role(run_full, gateway):
    out = run_full()
    roles = {u.role for u in out["llm_usage"]}
    assert roles == {"planner", "retrieval", "analyst", "reporter"}
    totals = gateway.ledger.totals(out["run_id"])
    assert totals["total_tokens"] == sum(u.total_tokens for u in out["llm_usage"])


def test_metrics_snapshot_has_nodes_api_and_llm(run_full, metrics):
    run_full()
    snap = metrics.snapshot()
    assert snap["nodes"]["execute_tool"]["success"] >= 3
    assert snap["api_calls"]["serp_organic"]["ok"] >= 1 and snap["api_calls"]["keyword_volume"]["ok"] == 1
    assert any(k.startswith("planner:") for k in snap["llm_calls"])


def test_mermaid_diagram_lists_all_nodes(graph):
    diagram = mermaid(graph)
    for node in EXPECTED_NODES | FALLBACK_NODES | {"retrieval_branch"}:
        assert node in diagram


def test_recheck_mode_runs_single_query(run_recheck):
    out = run_recheck("best seo software")
    assert out["plan_source"] == "recheck" and len(out["plan"].sub_queries) == 1
    assert len(out["planned_calls"]) == 3  # serp + ai check + volume
    assert len(out["records"]) == 1 and out["records"][0].query_text == "best seo software"
    assert out["status"] in {"completed", "partial"} and out["report"] is not None
    assert "query_planner" not in {t.node for t in out["node_traces"]}
