"""Deterministic parsers: raw DataForSEO payloads → typed fragments for the normalizer.

No LLM involvement here; this is pure extraction. Each parser raises `ParseError` on a
payload it cannot interpret so the normalizer can flag the record instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.dataforseo.domains import domain_from_url, domain_matches, normalize_domain


class ParseError(ValueError):
    pass


def _first_result(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        tasks = payload["tasks"]
        result = tasks[0]["result"]
        if not isinstance(result, list) or not result:
            raise ParseError("empty result list")
        first = result[0]
        if not isinstance(first, dict):
            raise ParseError("result[0] is not an object")
        return first
    except (KeyError, IndexError, TypeError) as exc:
        raise ParseError(f"unexpected payload shape: {exc!r}") from exc


@dataclass
class SerpParsed:
    domain_position: int | None
    competitor_positions: dict[str, int]
    top_domains: list[str]
    ai_overview_present: bool
    ai_overview_domains: list[str]
    ai_overview_cites_domain: bool


def parse_serp_organic(payload: dict[str, Any], *, profile_domain: str, competitors: list[str]) -> SerpParsed:
    result = _first_result(payload)
    items = result.get("items")
    if not isinstance(items, list):
        raise ParseError("items is not a list")

    domain_position: int | None = None
    competitor_positions: dict[str, int] = {}
    top_domains: list[str] = []
    ai_present = False
    ai_domains: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "organic":
            dom = normalize_domain(str(item.get("domain") or domain_from_url(item.get("url")) or ""))
            pos = item.get("rank_group") or item.get("rank_absolute")
            if not dom or not isinstance(pos, int):
                continue
            if len(top_domains) < 10:
                top_domains.append(dom)
            if domain_position is None and domain_matches(dom, profile_domain):
                domain_position = pos
            for comp in competitors:
                if comp not in competitor_positions and domain_matches(dom, comp):
                    competitor_positions[normalize_domain(comp)] = pos
        elif itype == "ai_overview":
            ai_present = True
            for ref in item.get("references") or []:
                if isinstance(ref, dict):
                    dom = ref.get("domain") or domain_from_url(ref.get("url"))
                    if dom:
                        ai_domains.append(normalize_domain(str(dom)))

    return SerpParsed(
        domain_position=domain_position,
        competitor_positions=competitor_positions,
        top_domains=top_domains,
        ai_overview_present=ai_present,
        ai_overview_domains=ai_domains,
        ai_overview_cites_domain=any(domain_matches(d, profile_domain) for d in ai_domains),
    )


@dataclass
class LlmParsed:
    cited_domains: list[str]
    domain_cited: bool
    competitors_cited: list[str]
    excerpt: str


def parse_llm_responses(payload: dict[str, Any], *, profile_domain: str, competitors: list[str]) -> LlmParsed:
    result = _first_result(payload)
    items = result.get("items")
    if not isinstance(items, list):
        raise ParseError("items is not a list")

    cited: list[str] = []
    text_parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for section in item.get("sections") or []:
            if not isinstance(section, dict):
                continue
            if section.get("text"):
                text_parts.append(str(section["text"]))
            for ann in section.get("annotations") or []:
                if isinstance(ann, dict):
                    dom = domain_from_url(ann.get("url"))
                    if dom and dom not in cited:
                        cited.append(dom)

    excerpt = " ".join(text_parts)[:500]
    domain_cited = any(domain_matches(d, profile_domain) for d in cited) or (
        profile_domain.split(".")[0].lower() in excerpt.lower()
    )
    competitors_cited = [normalize_domain(c) for c in competitors if any(domain_matches(d, c) for d in cited)]
    return LlmParsed(
        cited_domains=cited, domain_cited=domain_cited, competitors_cited=competitors_cited, excerpt=excerpt
    )


@dataclass
class VolumeParsed:
    by_keyword: dict[str, tuple[int | None, int | None]] = field(
        default_factory=dict
    )  # kw → (volume, competition_index)


def parse_keyword_volume(payload: dict[str, Any]) -> VolumeParsed:
    try:
        rows = payload["tasks"][0]["result"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ParseError(f"unexpected payload shape: {exc!r}") from exc
    if not isinstance(rows, list):
        raise ParseError("result is not a list")
    parsed = VolumeParsed()
    for row in rows:
        if not isinstance(row, dict) or not row.get("keyword"):
            continue
        vol = row.get("search_volume")
        comp = row.get("competition_index")
        parsed.by_keyword[str(row["keyword"]).lower()] = (
            int(vol) if isinstance(vol, int | float) else None,
            int(comp) if isinstance(comp, int | float) else None,
        )
    return parsed
