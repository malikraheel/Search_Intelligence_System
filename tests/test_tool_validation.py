"""Tool-call argument validation: the code must validate the LLM's arguments before any API call,
and must handle malformed / partial arguments without raising."""

from __future__ import annotations

import pytest

from app.tools.registry import TOOLS, to_openai_tools, validate_tool_call
from app.tools.schemas import KeywordVolumeInput, SerpOrganicInput


def test_openai_tool_definitions_have_schema_and_required_fields():
    tools = to_openai_tools()
    assert {t["function"]["name"] for t in tools} == set(TOOLS)
    serp = next(t for t in tools if t["function"]["name"] == "serp_organic")["function"]["parameters"]
    assert serp["required"] == ["keyword"]
    assert serp["properties"]["depth"]["maximum"] == 100
    assert serp["additionalProperties"] is False
    assert "AI Overview" in next(t for t in tools if t["function"]["name"] == "serp_organic")["function"]["description"]


def test_valid_dict_arguments_pass():
    out = validate_tool_call("serp_organic", {"keyword": "best seo software", "depth": 20})
    assert out.ok and isinstance(out.input, SerpOrganicInput)
    assert out.input.depth == 20 and out.input.location_code == 2840


def test_valid_json_string_arguments_are_parsed():
    out = validate_tool_call("llm_responses", '{"user_prompt": "best seo tool?", "web_search": true}')
    assert out.ok and out.input.user_prompt == "best seo tool?"


def test_missing_required_field_is_reported_not_raised():
    out = validate_tool_call("serp_organic", {"depth": 10})
    assert not out.ok
    assert any("keyword" in e and "required" in e.lower() for e in out.errors)


def test_partial_arguments_get_profile_defaults_for_missing_keys_only():
    out = validate_tool_call(
        "serp_organic", {"keyword": "x", "language_code": "de"}, defaults={"location_code": 2276, "language_code": "en"}
    )
    assert out.ok
    assert out.input.location_code == 2276  # filled
    assert out.input.language_code == "de"  # LLM's explicit value kept


def test_hallucinated_extra_field_is_rejected():
    out = validate_tool_call("serp_organic", {"keyword": "x", "country": "US"})
    assert not out.ok and any("country" in e for e in out.errors)


@pytest.mark.parametrize(
    "bad", [{"keyword": "x", "depth": 7}, {"keyword": "x", "depth": 500}, {"keyword": "", "depth": 10}]
)
def test_out_of_range_or_empty_values_rejected(bad):
    assert not validate_tool_call("serp_organic", bad).ok


def test_malformed_json_string_is_reported():
    out = validate_tool_call("serp_organic", '{"keyword": "x", ')
    assert not out.ok and "not valid JSON" in out.errors[0]


@pytest.mark.parametrize("raw", [None, "", "[1,2]", 42])
def test_non_object_arguments_reported(raw):
    out = validate_tool_call("serp_organic", raw)
    assert not out.ok and out.errors


def test_unknown_tool_name_reported():
    out = validate_tool_call("call_dataforseo", {"anything": 1})
    assert not out.ok and "unknown tool" in out.errors[0]


def test_keyword_volume_dedupes_and_cleans():
    out = validate_tool_call("keyword_volume", {"keywords": ["SEO tool", "seo  tool", " content optimizer "]})
    assert out.ok and isinstance(out.input, KeywordVolumeInput)
    assert out.input.keywords == ["SEO tool", "content optimizer"]


def test_keyword_volume_rejects_empty_keyword():
    assert not validate_tool_call("keyword_volume", {"keywords": ["ok", "   "]}).ok
