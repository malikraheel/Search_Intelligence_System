"""Opportunity scoring — pure, explainable functions. The LLM never computes scores.

vol_norm       = clamp(log10(1+volume) / log10(1+100_000), 0, 1)        (None → 0.3)
difficulty     = clamp(0.6*competition_index + 0.4*min(competitors_in_top10,5)*20, 0, 100)
visibility_gap = 1.0 not visible | clamp((pos-1)/20, 0, 1)*0.5 visible | 0.6 unknown
ai_gap         = 1.0 not cited | 0.2 cited | 0.6 unknown
opportunity    = 0.35*vol_norm + 0.25*(1-difficulty/100) + 0.25*visibility_gap + 0.15*ai_gap
"""

from __future__ import annotations

import math

from app.domain.models import ScoreComponents

VOLUME_SATURATION = 100_000
W_VOLUME, W_DIFFICULTY, W_VISIBILITY, W_AI = 0.35, 0.25, 0.25, 0.15


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def volume_norm(search_volume: int | None) -> float:
    if search_volume is None:
        return 0.3
    return clamp(math.log10(1 + max(0, search_volume)) / math.log10(1 + VOLUME_SATURATION), 0.0, 1.0)


def competitive_difficulty(competition_index: int | None, competitors_in_top10: int) -> int:
    ci = 50 if competition_index is None else clamp(competition_index, 0, 100)
    raw = 0.6 * ci + 0.4 * min(max(competitors_in_top10, 0), 5) * 20
    return int(round(clamp(raw, 0, 100)))


def visibility_gap(domain_visible: bool | None, position: int | None) -> float:
    if domain_visible is None:
        return 0.6
    if not domain_visible:
        return 1.0
    pos = position or 1
    return clamp((pos - 1) / 20, 0.0, 1.0) * 0.5


def ai_gap(ai_cited: bool | None) -> float:
    if ai_cited is None:
        return 0.6
    return 0.2 if ai_cited else 1.0


def opportunity_score(
    *,
    search_volume: int | None,
    difficulty: int,
    domain_visible: bool | None,
    position: int | None,
    ai_cited: bool | None,
) -> tuple[float, ScoreComponents]:
    comps = ScoreComponents(
        vol_norm=round(volume_norm(search_volume), 4),
        difficulty_norm=round(1 - clamp(difficulty, 0, 100) / 100, 4),
        visibility_gap=round(visibility_gap(domain_visible, position), 4),
        ai_gap=round(ai_gap(ai_cited), 4),
    )
    score = (
        W_VOLUME * comps.vol_norm
        + W_DIFFICULTY * comps.difficulty_norm
        + W_VISIBILITY * comps.visibility_gap
        + W_AI * comps.ai_gap
    )
    return round(clamp(score, 0.0, 1.0), 4), comps
