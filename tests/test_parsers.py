"""Deterministic extraction from DataForSEO payloads (mock fixtures share the live shape)."""

from __future__ import annotations

import pytest

from app.dataforseo import mock_data
from app.dataforseo.domains import domain_matches, normalize_domain
from app.dataforseo.parsers import ParseError, parse_keyword_volume, parse_llm_responses, parse_serp_organic


def test_domain_helpers():
    assert normalize_domain("https://www.SurferSEO.com/path") == "surferseo.com"
    assert domain_matches("blog.surferseo.com", "surferseo.com")
    assert not domain_matches("notsurferseo.com", "surferseo.com")
    assert not domain_matches(None, "surferseo.com")


def test_parse_serp_finds_positions_and_ai_overview():
    payload = mock_data.serp_organic_response(
        keyword="brand: surfer seo",
        location_code=2840,
        language_code="en",
        depth=10,
        profile_domain="surferseo.com",
        competitors=["clearscope.io", "frase.io"],
    )
    parsed = parse_serp_organic(payload, profile_domain="surferseo.com", competitors=["clearscope.io", "frase.io"])
    assert parsed.domain_position == 1
    assert len(parsed.top_domains) == 10
    assert all(v >= 1 for v in parsed.competitor_positions.values())
    assert isinstance(parsed.ai_overview_present, bool)


def test_parse_serp_absent_domain_gives_none():
    payload = mock_data.serp_organic_response(
        keyword="totally unrelated",
        location_code=2840,
        language_code="en",
        depth=10,
        profile_domain="nobody.example",
        competitors=[],
    )
    # force removal of the profile domain to assert the None path deterministically
    for item in payload["tasks"][0]["result"][0]["items"]:
        if item.get("domain") == "nobody.example":
            item["domain"] = "other.example"
    parsed = parse_serp_organic(payload, profile_domain="nobody.example", competitors=[])
    assert parsed.domain_position is None


def test_parse_llm_citations():
    payload = mock_data.llm_responses_response(
        prompt="best seo tools", model_name="gpt-4o-mini", profile_domain="surferseo.com", competitors=["frase.io"]
    )
    parsed = parse_llm_responses(payload, profile_domain="surferseo.com", competitors=["frase.io"])
    assert parsed.cited_domains and parsed.excerpt
    assert parsed.domain_cited == ("surferseo.com" in parsed.cited_domains or "surferseo" in parsed.excerpt.lower())


def test_parse_volume():
    payload = mock_data.keyword_volume_response(keywords=["SEO tool", "x"], location_code=2840, language_code="en")
    parsed = parse_keyword_volume(payload)
    assert set(parsed.by_keyword) == {"seo tool", "x"}
    vol, comp = parsed.by_keyword["seo tool"]
    assert vol > 0 and 0 <= comp <= 100


@pytest.mark.parametrize(
    "bad", [{}, {"tasks": []}, {"tasks": [{"result": []}]}, {"tasks": [{"result": [{"items": "nope"}]}]}]
)
def test_parse_serp_malformed_raises_parse_error(bad):
    with pytest.raises(ParseError):
        parse_serp_organic(bad, profile_domain="a.com", competitors=[])
