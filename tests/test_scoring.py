from __future__ import annotations

from app.services.scoring import competitive_difficulty, opportunity_score, volume_norm


def test_volume_norm_bounds_and_monotonic():
    assert volume_norm(None) == 0.3
    assert volume_norm(0) == 0.0
    assert volume_norm(100_000) == 1.0 and volume_norm(10_000_000) == 1.0
    assert volume_norm(100) < volume_norm(1000) < volume_norm(10_000)


def test_difficulty_bounds():
    assert competitive_difficulty(None, 0) == 30  # 0.6*50
    assert competitive_difficulty(100, 5) == 100
    assert competitive_difficulty(0, 0) == 0
    assert 0 <= competitive_difficulty(73, 2) <= 100


def test_opportunity_prefers_invisible_high_volume_low_difficulty():
    hi, comps = opportunity_score(
        search_volume=50_000, difficulty=20, domain_visible=False, position=None, ai_cited=False
    )
    lo, _ = opportunity_score(search_volume=50, difficulty=90, domain_visible=True, position=1, ai_cited=True)
    assert 0 <= lo < hi <= 1
    assert comps.visibility_gap == 1.0 and comps.ai_gap == 1.0


def test_unknowns_use_neutral_defaults():
    s, comps = opportunity_score(search_volume=None, difficulty=50, domain_visible=None, position=None, ai_cited=None)
    assert comps.vol_norm == 0.3 and comps.visibility_gap == 0.6 and comps.ai_gap == 0.6
    assert 0 < s < 1


def test_visible_position_matters():
    p1, _ = opportunity_score(search_volume=1000, difficulty=50, domain_visible=True, position=1, ai_cited=None)
    p15, _ = opportunity_score(search_volume=1000, difficulty=50, domain_visible=True, position=15, ai_cited=None)
    assert p15 > p1
