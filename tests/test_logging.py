"""Structured logging: redaction of secrets, truncation, and context propagation."""

from __future__ import annotations

import json

import structlog

from app.observability.logging import bind_context, configure_logging, get_logger, redact


def test_redact_masks_credentials_but_keeps_token_counts():
    out = redact(
        {
            "api_key": "sk-123",
            "OPENAI_API_KEY": "sk-456",
            "password": "pw",
            "Authorization": "Bearer x",
            "access_token": "abc",
            "login": "user@example.com",
            "tokens": 120,
            "total_tokens": 300,
            "prompt_tokens": 100,
            "nested": {"dataforseo_password": "x", "keyword": "ok"},
            "items": [{"secret": "x"}, "plain"],
        }
    )
    assert out["api_key"] == out["OPENAI_API_KEY"] == out["password"] == out["Authorization"] == "***REDACTED***"
    assert out["access_token"] == out["login"] == "***REDACTED***"
    assert out["tokens"] == 120 and out["total_tokens"] == 300 and out["prompt_tokens"] == 100
    assert out["nested"] == {"dataforseo_password": "***REDACTED***", "keyword": "ok"}
    assert out["items"] == [{"secret": "***REDACTED***"}, "plain"]


def test_redact_truncates_long_strings_and_handles_models():
    long = "x" * 2000
    out = redact({"payload": long})
    assert out["payload"].startswith("x" * 500) and "+1500 chars" in out["payload"]

    from app.domain.models import LLMUsage

    assert redact(LLMUsage(role="planner", model="m", total_tokens=5))["total_tokens"] == 5


def test_json_log_line_carries_bound_context(capsys):
    configure_logging("INFO", "json")
    try:
        with bind_context(run_id="r-1", node="query_planner"):
            get_logger("test").info("node.end", duration_ms=3.2, api_key="should-hide", tokens=42)
        line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert line["event"] == "node.end" and line["run_id"] == "r-1" and line["node"] == "query_planner"
        assert line["component"] == "test" and line["level"] == "info" and line["timestamp"].endswith("Z")
        assert line["api_key"] == "***REDACTED***" and line["tokens"] == 42
        # context is released after the block
        get_logger("test").info("after")
        after = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "run_id" not in after
    finally:
        configure_logging("WARNING", "json")
        structlog.contextvars.clear_contextvars()
