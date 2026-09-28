"""Unit tests for the v3 desks: point-in-time discipline is the whole game.

Every test below constructs a FakeFD whose responses straddle `as_of` and
asserts the desk only sees the past. A desk that peeks at the future fails.

Run: ./.venv/bin/python tests/test_desks.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.research.desks import (  # noqa: E402
    fundamentals_scores,
    macro_factor,
    pead_scores,
    sentiment_scores,
)
from fund.decision.fusion import (  # noqa: E402
    anonymize_scores,
    blend_regime,
    deanonymize_convictions,
    fuse_desks,
)
from fund.research.schema import AgentScore  # noqa: E402

AS_OF = date(2024, 3, 1)


class FakeFD:
    """Scripted Financial Datasets stand-in."""

    def __init__(self):
        self.metrics_rows: dict[str, list[dict]] = {}
        self.earnings_rows: dict[str, list[dict]] = {}
        self.news_rows: dict[str, list[dict]] = {}
        self.curves: list[dict] = []
        self.cpi: list[dict] = []

    def metrics(self, ticker, period="quarterly", limit=12):
        return self.metrics_rows.get(ticker, [])[:limit]

    def earnings(self, ticker, limit=8):
        return self.earnings_rows.get(ticker, [])[:limit]

    def news(self, ticker, limit=10):
        return self.news_rows.get(ticker, [])[:limit]

    def yield_curve_history(self, limit=60):
        return self.curves[:limit]

    def inflation_history(self, limit=36):
        return self.cpi[:limit]


def _tickers(n=12):
    return [f"STK{i:02d}" for i in range(n)]


# ------------------------------------------------------- fundamentals gating
def test_fundamentals_uses_only_filed_metrics():
    fd = FakeFD()
    for i, t in enumerate(_tickers()):
        # two rows: one filed before as_of (cheap), one filed AFTER (expensive).
        # A lookahead leak would see the expensive row and flip the ranking.
        fd.metrics_rows[t] = [
            {"filing_date": "2024-04-15",  # AFTER as_of: must be invisible
             "price_to_earnings_ratio": 50.0,
             "free_cash_flow_yield": 0.01,
             "return_on_equity": 0.05,
             "gross_margin": 0.2,
             "earnings_per_share_growth": 0.0},
            {"filing_date": "2024-02-01",  # before as_of: the real past
             "price_to_earnings_ratio": 10.0 + i,  # STK00 cheapest
             "free_cash_flow_yield": 0.08,
             "return_on_equity": 0.20,
             "gross_margin": 0.4,
             "earnings_per_share_growth": 0.1},
        ]
    out = fundamentals_scores(fd, _tickers(), AS_OF, {})
    assert len(out) == 12, f"expected 12, got {len(out)}"
    # cheapest on filed metrics (STK00, PE 10) must outrank STK11 (PE 21)
    assert out["STK00"][0] > out["STK11"][0], \
        f"lookahead leak? STK00={out['STK00'][0]:.3f} STK11={out['STK11'][0]:.3f}"
    print("ok: fundamentals gated on filing_date (no lookahead)")


def test_fundamentals_skips_unfiled_tickers():
    fd = FakeFD()
    fd.metrics_rows["AAA"] = [{"filing_date": "2025-01-01",
                               "price_to_earnings_ratio": 5.0}]
    out = fundamentals_scores(fd, ["AAA"], AS_OF, {})
    assert out == {}, "ticker with nothing filed as of T must emit no score"
    print("ok: unfiled ticker emits no view")


# ------------------------------------------------------------------ pead
def test_pead_fires_on_fresh_beat_not_stale():
    fd = FakeFD()
    fd.earnings_rows["FRESH"] = [
        {"filing_date": "2024-02-20", "eps_surprise": "BEAT",
         "eps_surprise_pct": 15.0}]
    fd.earnings_rows["STALE"] = [
        {"filing_date": "2023-12-01", "eps_surprise": "BEAT",
         "eps_surprise_pct": 50.0}]  # huge but 90d old: not a signal
    fd.earnings_rows["FUTURE"] = [
        {"filing_date": "2024-03-10", "eps_surprise": "MISS",
         "eps_surprise_pct": 30.0}]  # after as_of: invisible
    fd.earnings_rows["MISS"] = [
        {"filing_date": "2024-02-25", "eps_surprise": "MISS",
         "eps_surprise_pct": 12.0}]
    # pad so z-scoring has enough points (tiny surprises -> weak views)
    for i in range(10):
        fd.earnings_rows[f"PAD{i:02d}"] = [
            {"filing_date": "2024-02-15", "eps_surprise": "BEAT",
             "eps_surprise_pct": 0.5 if i % 2 == 0 else -0.5}]
    tickers = ["FRESH", "STALE", "FUTURE", "MISS"] + [f"PAD{i:02d}" for i in range(10)]
    out = pead_scores(fd, tickers, AS_OF, {"pead_window_days": 21})
    assert "FRESH" in out and out["FRESH"][0] > 0, "fresh BEAT must be bullish"
    assert "MISS" in out and out["MISS"][0] < 0, "fresh MISS must be bearish"
    assert "STALE" not in out, "90-day-old filing must not fire"
    assert "FUTURE" not in out, "filing after as_of must be invisible"
    print("ok: PEAD fires on fresh events only (filing_date <= as_of, in-window)")


# --------------------------------------------------------------- sentiment
def test_sentiment_ignores_future_news():
    fd = FakeFD()
    tickers = ["UP", "DN"] + [f"Q{i:02d}" for i in range(10)]
    idx = pd.date_range("2023-06-01", "2024-03-01", freq="B")
    closes = pd.DataFrame(
        {t: 100 + np.arange(len(idx)) * (0.1 if t == "UP" else 0.0)
         for t in tickers}, index=idx)
    # baseline: 4 old items each (enough history for the z-score, no signal)
    for t in tickers:
        fd.news_rows[t] = [
            {"published_at": f"2023-0{d}-15", "title": f"old{d}"} for d in range(6, 10)
        ]
    # UP: heavy RECENT news (before as_of) + rising price -> positive
    fd.news_rows["UP"] += [
        {"published_at": f"2024-02-{d:02d}", "title": f"n{d}"} for d in range(20, 28)
    ]
    # DN: heavy FUTURE news (after as_of) -> must be invisible
    fd.news_rows["DN"] = [
        {"published_at": f"2024-03-{d:02d}", "title": f"f{d}"} for d in range(5, 12)
    ] + [{"published_at": "2023-08-01", "title": "old"}]
    out = sentiment_scores(fd, tickers, AS_OF, closes, {})
    assert "UP" in out and out["UP"][0] > 0, f"UP should be positive: {out.get('UP')}"
    assert "DN" not in out, f"DN's news is all future: {out.get('DN')}"
    # padding tickers: no abnormal attention -> near-zero scores
    for t in [f"Q{i:02d}" for i in range(10)]:
        assert abs(out[t][0]) < 0.5, f"{t} should be neutral: {out[t][0]}"
    print("ok: sentiment gated on published_at <= as_of")


# ------------------------------------------------------------------- macro
def test_macro_factor_neutral_without_data():
    fd = FakeFD()
    f = macro_factor(fd, AS_OF, {})
    assert f == 1.0, f"no macro data must not gate the book (got {f})"
    print("ok: macro factor = 1.0 when data is absent")


def test_blend_regime_bounds():
    assert blend_regime(0.5, 1.0, blend=0.0, floor=0.5) == 0.5
    assert blend_regime(0.5, 1.0, blend=1.0, floor=0.5) == 1.0
    assert blend_regime(0.5, 1.0, blend=0.5, floor=0.5) == 0.75
    assert blend_regime(0.2, 0.2, blend=0.5, floor=0.5) == 0.5  # floored
    print("ok: regime blending bounded and floored")


# ------------------------------------------------------- fusion + anonymize
def _scores():
    s = []
    for i, t in enumerate(_tickers()):
        s.append(AgentScore(t, "trend", float(i) / 12, 0.9, f"trend {t}", AS_OF))
        s.append(AgentScore(t, "fundamentals", 1 - float(i) / 12, 0.8,
                            f"fund {t}", AS_OF))
    return s


def test_fuse_desks_two_level():
    conv = fuse_desks(_scores(),
                      {"technicals": 0.4, "fundamentals": 0.3,
                       "sentiment": 0.15, "macro": 0.15},
                      regime_factor=1.0)
    assert set(conv) == set(_tickers())
    assert all(0 < v < 1 for v in conv.values())
    # STK11 tops trend but bottoms fundamentals; STK00 the reverse.
    # With technicals 0.4 > fundamentals 0.3, STK11 should win at f=1.0.
    assert conv["STK11"] > conv["STK00"], \
        f"desk weights not applied: {conv['STK11']:.3f} vs {conv['STK00']:.3f}"
    print("ok: desk fusion applies mandate weights")


def test_anonymize_roundtrip():
    scores = _scores()
    blind, mapping = anonymize_scores(scores)
    assert set(mapping) == set(_tickers())
    assert len(set(mapping.values())) == 12  # bijective
    for s in blind:
        assert s.ticker.startswith("T-"), s.ticker
        assert s.as_of is None, "dates must be withheld"
        for t in _tickers():
            assert t not in s.rationale, f"ticker leaked in rationale: {s.rationale}"
    # simulate the judge ruling on blind labels, then map back
    blind_conv = {lbl: 0.1 * i for i, lbl in enumerate(sorted(mapping.values()))}
    conv = deanonymize_convictions(blind_conv, mapping)
    assert set(conv) == set(_tickers())
    print("ok: anonymize/deanonymize round-trips, no identity leaks")


if __name__ == "__main__":
    test_fundamentals_uses_only_filed_metrics()
    test_fundamentals_skips_unfiled_tickers()
    test_pead_fires_on_fresh_beat_not_stale()
    test_sentiment_ignores_future_news()
    test_macro_factor_neutral_without_data()
    test_blend_regime_bounds()
    test_fuse_desks_two_level()
    test_anonymize_roundtrip()
    print("ALL DESK TESTS PASSED")
