"""Simulated failures: retries that recover, retries that exhaust → fallback paths, degraded statuses."""

from __future__ import annotations

from app.gateway import Role
from app.gateway.errors import ProviderNonRetryable
from app.gateway.providers.fake import json_response, text_response, tool_call_response
from tests.fakes import model_for


def _nodes(out) -> set[str]:
    return {t.node for t in out["node_traces"]}


# ── DataForSEO failures ────────────────────────────────────────────────────────
def test_transient_api_errors_are_retried_and_run_completes(run_full):
    out = run_full(chaos="serp_organic:429,503,ok", max_concurrency=1)
    assert out["status"] == "completed"
    retried = [t for t in out["node_traces"] if t.node == "execute_tool" and t.retry_count == 2]
    assert len(retried) == 1 and retried[0].api_calls == 3
    assert any(r.retry_count == 2 for r in out["raw_results"])


def test_exhausted_retries_route_to_retrieval_fallback_and_partial_status(run_full):
    out = run_full(chaos="llm_responses:500,500,500,500", max_concurrency=1)  # 4 attempts → exhausted
    assert out["status"] == "partial"
    assert "retrieval_fallback" in _nodes(out)
    degraded = [r for r in out["raw_results"] if r.status == "degraded"]
    assert len(degraded) == 1 and degraded[0].tool == "llm_responses" and degraded[0].error.retryable
    flagged = [r for r in out["records"] if r.error_flags]
    assert len(flagged) == 1 and flagged[0].error_flags == ["llm_responses_failed:http"]
    assert any("llm_responses" in d for d in out["degradation"])
    assert len(out["records"]) == len(out["plan"].sub_queries)  # the query itself survives (SERP + volume ok)
    assert out["report"].report_json["status"] == "partial"


def test_non_retryable_api_error_fails_fast_into_fallback(run_full, metrics):
    out = run_full(chaos="serp_organic:401", max_concurrency=1)
    assert out["status"] == "partial"
    degraded = [r for r in out["raw_results"] if r.status == "degraded"][0]
    assert degraded.error.http_status == 401 and not degraded.error.retryable
    assert metrics.snapshot()["api_calls"]["serp_organic"]["non_retryable_error"] == 1
    rec = next(r for r in out["records"] if r.query_text == degraded.query_text)
    assert rec.visibility_status == "unknown" and rec.domain_visible is None


def test_every_tool_failing_yields_degraded_report_and_failed_status(run_full):
    script = ";".join(f"{t}:" + ",".join(["500"] * 40) for t in ("serp_organic", "llm_responses", "keyword_volume"))
    out = run_full(chaos=script, max_concurrency=1)
    assert out["status"] == "failed"
    assert "degraded_report" in _nodes(out) and "analyze" not in _nodes(out)
    assert out["records"] == [] and out["dropped_records"] == len(out["plan"].sub_queries)
    assert all(r.status == "degraded" for r in out["raw_results"])
    assert "failed" in out["report"].summary


def test_malformed_payload_is_flagged_not_fatal(run_full):
    out = run_full(chaos="serp_organic:malformed", max_concurrency=1)
    assert out["status"] in {"completed", "partial"}
    assert any("serp_organic_parse_error" in r.error_flags for r in out["records"])
    assert any(e.kind == "parse_error" for e in out["errors"])


# ── planner failures ───────────────────────────────────────────────────────────
def test_planner_unavailable_uses_deterministic_fallback_plan(run_full, fake_llm):
    fake_llm.enqueue(model_for(Role.PLANNER), ProviderNonRetryable("401 invalid api key"))
    out = run_full()
    assert out["plan_source"] == "fallback" and "fallback_plan" in _nodes(out)
    assert "plan_repair" not in _nodes(out)
    assert out["status"] == "partial" and any(d.startswith("planner_fallback") for d in out["degradation"])
    assert len(out["records"]) >= 4 and out["analysis"] is not None


def test_planner_garbage_json_triggers_repair_then_fallback(run_full, fake_llm):
    m = model_for(Role.PLANNER)
    fake_llm.enqueue(m, text_response("I cannot help with that."), text_response("still not json"))
    out = run_full()
    assert out["plan_source"] == "fallback"
    assert any(e.node == "query_planner" for e in out["errors"])


def test_invalid_plan_is_repaired_by_second_llm_call(run_full, fake_llm):
    m = model_for(Role.PLANNER)
    bad = {"sub_queries": [{"query_text": "best seo software", "intent": "category"}], "rationale": "too few, no brand"}
    fake_llm.enqueue(m, json_response(bad))  # first attempt fails business validation → plan_repair → smart_llm
    out = run_full()
    assert "plan_repair" in _nodes(out) and out["plan_source"] == "repaired"
    assert out["status"] == "completed"


# ── tool-call argument validation in the graph ────────────────────────────────
def test_bad_tool_arguments_are_repaired_once(run_full, fake_llm):
    m = model_for(Role.RETRIEVAL)
    fake_llm.enqueue(m, tool_call_response("serp_organic", {"keyword": "x", "depth": 7, "country": "US"}))
    out = run_full(max_concurrency=1)
    validators = [t for t in out["node_traces"] if t.node == "tool_arg_validator"]
    rejected = [t for t in validators if t.output_summary.get("validation_errors")]
    assert rejected, "first proposal should have been rejected"
    errs = " ".join(rejected[0].output_summary["validation_errors"])
    assert "depth" in errs and "country" in errs
    assert sum(1 for t in out["node_traces"] if t.node == "retrieval_agent") == len(out["planned_calls"]) + 1
    assert out["status"] == "completed"


def test_persistently_bad_tool_arguments_fall_back(run_full, fake_llm):
    m = model_for(Role.RETRIEVAL)
    fake_llm.enqueue(m, tool_call_response("serp_organic", '{"keyword": '), tool_call_response("nope_tool", {}))
    out = run_full(max_concurrency=1)
    assert "retrieval_fallback" in _nodes(out) and out["status"] == "partial"
    degraded = [r for r in out["raw_results"] if r.status == "degraded"][0]
    assert degraded.error.kind == "tool_args_invalid"


def test_wrong_tool_for_step_is_rejected(run_full, fake_llm):
    m = model_for(Role.RETRIEVAL)
    fake_llm.enqueue(m, tool_call_response("llm_responses", {"user_prompt": "hi"}))  # step expects a different tool
    out = run_full(max_concurrency=1)
    rejected = [
        t for t in out["node_traces"] if t.node == "tool_arg_validator" and t.output_summary.get("validation_errors")
    ]
    assert rejected and "expected tool" in rejected[0].output_summary["validation_errors"][0]


def test_retrieval_llm_unavailable_falls_back_without_retrying_llm(run_full, fake_llm):
    m = model_for(Role.RETRIEVAL)
    fake_llm.enqueue(m, ProviderNonRetryable("401"))
    out = run_full(max_concurrency=1)
    assert out["status"] == "partial"
    degraded = [r for r in out["raw_results"] if r.status == "degraded"]
    assert len(degraded) == 1 and "LLM unavailable" in degraded[0].error.message


# ── analysis / report failures ─────────────────────────────────────────────────
def test_analyst_failure_uses_rule_based_fallback(run_full, fake_llm):
    fake_llm.enqueue(model_for(Role.ANALYST), ProviderNonRetryable("400 bad request"))
    out = run_full()
    assert "analyze_fallback" in _nodes(out) and out["analysis_source"] == "fallback"
    assert out["status"] == "partial" and out["analysis"].insights and out["analysis"].recommendations
    assert all(r.priority in {"high", "medium", "low"} for r in out["analysis"].recommendations)


def test_reporter_failure_keeps_deterministic_summary(run_full, fake_llm):
    fake_llm.enqueue(model_for(Role.REPORTER), ProviderNonRetryable("500-ish but non retryable"))
    out = run_full()
    assert out["status"] == "completed"  # reporter prose is optional
    assert "visibility report" in out["report"].summary
    assert any(e.node == "report" for e in out["errors"])
