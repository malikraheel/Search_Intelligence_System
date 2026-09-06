"""Deterministic, realistic DataForSEO response fixtures for mock mode.

Responses mirror the real API envelope (`tasks[].result[].items[]`) so the normalizer
exercises the same code path in mock and live modes. Everything is seeded from the keyword,
so the same query always yields the same "SERP" — reproducible demos and tests.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime
from typing import Any

FILLER_DOMAINS = [
    "g2.com",
    "capterra.com",
    "reddit.com",
    "forbes.com",
    "techradar.com",
    "zapier.com",
    "hubspot.com",
    "medium.com",
    "youtube.com",
    "trustradius.com",
    "softwareadvice.com",
    "pcmag.com",
]


def _seed(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:12], 16))


def _envelope(path: list[str], result: list[dict[str, Any]], *, cost: float) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S +00:00")
    return {
        "version": "0.1.20250101",
        "status_code": 20000,
        "status_message": "Ok.",
        "time": "0.4213 sec.",
        "cost": cost,
        "tasks_count": 1,
        "tasks_error": 0,
        "tasks": [
            {
                "id": hashlib.md5(repr(result).encode()).hexdigest(),
                "status_code": 20000,
                "status_message": "Ok.",
                "time": "0.4001 sec.",
                "cost": cost,
                "result_count": len(result),
                "path": path,
                "data": {"api": path[1], "function": path[-1]},
                "result": result,
                "_mock": True,
                "_datetime": now,
            }
        ],
    }


def _ranked_domains(
    rng: random.Random, keyword: str, profile_domain: str, competitors: list[str], depth: int
) -> list[str]:
    pool = FILLER_DOMAINS[:]
    rng.shuffle(pool)
    ranked = pool[:depth]
    # competitors: 1-3 of them appear somewhere in the top results
    for comp in competitors[: rng.randint(1, min(3, max(1, len(competitors))))] if competitors else []:
        ranked[rng.randrange(0, min(depth, 10))] = comp
    # profile domain: visible ~60% of the time; brand queries almost always rank #1
    roll = rng.random()
    if "brand:" in keyword or roll < 0.6:
        position = 0 if "brand:" in keyword else rng.randrange(0, min(depth, 12))
        if position < depth:
            ranked[position] = profile_domain
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    while len(out) < depth:
        out.append(f"site{len(out)}.example.com")
    return out


def serp_organic_response(
    *, keyword: str, location_code: int, language_code: str, depth: int, profile_domain: str, competitors: list[str]
) -> dict[str, Any]:
    rng = _seed("serp", keyword.lower())
    ranked = _ranked_domains(rng, keyword.lower(), profile_domain, competitors, depth)
    items: list[dict[str, Any]] = []
    rank_absolute = 1

    if rng.random() < 0.7:  # AI Overview present
        ref_pool = [d for d in ranked[:8] if d != profile_domain]
        refs = rng.sample(ref_pool, k=min(3, len(ref_pool)))
        if rng.random() < 0.4:
            refs.insert(rng.randrange(0, len(refs) + 1), profile_domain)
        items.append(
            {
                "type": "ai_overview",
                "rank_group": 1,
                "rank_absolute": rank_absolute,
                "position": "left",
                "items": [{"type": "ai_overview_element", "title": None, "text": f"Overview of {keyword}: …"}],
                "references": [
                    {
                        "type": "ai_overview_reference",
                        "source": d.split(".")[0].title(),
                        "domain": d,
                        "url": f"https://{d}/{keyword.replace(' ', '-')}",
                        "title": f"{keyword.title()} — {d}",
                    }
                    for d in refs
                ],
            }
        )
        rank_absolute += 1

    for group, domain in enumerate(ranked, start=1):
        items.append(
            {
                "type": "organic",
                "rank_group": group,
                "rank_absolute": rank_absolute,
                "position": "left",
                "domain": domain,
                "title": f"{keyword.title()} | {domain}",
                "url": f"https://{domain}/{keyword.replace(' ', '-')}",
                "description": f"Everything about {keyword} from {domain}.",
            }
        )
        rank_absolute += 1

    result = {
        "keyword": keyword,
        "type": "organic",
        "se_domain": "google.com",
        "location_code": location_code,
        "language_code": language_code,
        "check_url": f"https://www.google.com/search?q={keyword.replace(' ', '+')}",
        "datetime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S +00:00"),
        "item_types": sorted({i["type"] for i in items}),
        "se_results_count": rng.randint(500_000, 250_000_000),
        "items_count": len(items),
        "items": items,
    }
    return _envelope(["v3", "serp", "google", "organic", "live", "advanced"], [result], cost=0.002)


def llm_responses_response(
    *, prompt: str, model_name: str, profile_domain: str, competitors: list[str]
) -> dict[str, Any]:
    rng = _seed("llm", prompt.lower())
    cited = rng.sample(FILLER_DOMAINS, k=2)
    cited += rng.sample(competitors, k=min(len(competitors), rng.randint(0, 2))) if competitors else []
    if rng.random() < 0.45:
        cited.append(profile_domain)
    rng.shuffle(cited)
    text = f"Here are strong options for '{prompt}': " + ", ".join(d.split(".")[0].title() for d in cited) + "."
    result = {
        "model_name": model_name,
        "input_tokens": 40 + len(prompt) // 4,
        "output_tokens": 120,
        "web_search": True,
        "money_spent": 0.0021,
        "datetime": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S +00:00"),
        "items": [
            {
                "type": "message",
                "sections": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {"type": "url_citation", "title": f"{d} — overview", "url": f"https://{d}/"} for d in cited
                        ],
                    }
                ],
            }
        ],
    }
    return _envelope(["v3", "ai_optimization", "chat_gpt", "llm_responses", "live"], [result], cost=0.0021)


def keyword_volume_response(*, keywords: list[str], location_code: int, language_code: str) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    for kw in keywords:
        rng = _seed("volume", kw.lower())
        words = len(kw.split())
        base = rng.choice([90, 320, 1300, 4400, 12100, 33100, 74000])
        volume = max(10, int(base / max(1, words - 1)))
        comp_idx = rng.randint(15, 95)
        result.append(
            {
                "keyword": kw,
                "spell": None,
                "location_code": location_code,
                "language_code": language_code,
                "search_partners": False,
                "competition": "HIGH" if comp_idx > 66 else "MEDIUM" if comp_idx > 33 else "LOW",
                "competition_index": comp_idx,
                "search_volume": volume,
                "low_top_of_page_bid": round(rng.uniform(0.5, 4.0), 2),
                "high_top_of_page_bid": round(rng.uniform(4.0, 20.0), 2),
                "cpc": round(rng.uniform(1.0, 12.0), 2),
                "monthly_searches": [
                    {"year": 2026, "month": m, "search_volume": int(volume * rng.uniform(0.8, 1.2))}
                    for m in range(1, 13)
                ],
            }
        )
    return _envelope(["v3", "keywords_data", "google_ads", "search_volume", "live"], result, cost=0.05)
