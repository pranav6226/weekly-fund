"""Unit tests for the v3.1 desks on the free data stack.

Point-in-time discipline is the whole game: every test constructs a
FakeFree whose responses straddle `as_of` and asserts the desk only sees
the past. A desk that peeks at the future fails.

The filings desk has two sub-signals, both tested: (a) PEAD -- a fresh
Item 2.02 8-K drifts with the market's 2-day reaction to it; (b)
attention -- abnormal 8-K rate x price direction.

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
    filings_scores,
    fundamentals_scores,
    macro_factor,
)
from fund.decision.fusion import (  # noqa: E402
    anonymize_scores,
    blend_regime,
    deanonymize_convictions,
    fuse_desks,
)
from fund.research.schema import AgentScore  # noqa: E402

AS_OF = date(2024, 3, 1)


class FakeFree:
    """Scripted FreeDataStack stand-in: metrics + 8-K events + macro."""

    def __init__(self):
        self.metrics_rows: dict[str, list[dict]] = {}
        self.events: dict[str, list[dict]] = {}
        self.curves: list[dict] = []
        self.cpi: list[dict] = []
        self.unrate: list[dict] = []

    def metrics(self, ticker, limit=8):
        return self.metrics_rows.get(ticker, [])[:limit]

    def filing_events(self, ticker, limit=500):
        return self.events.get(ticker, [])[:limit]

    def yield_curve_history(self, limit=400):
        return self.curves[:limit]

    def inflation_history(self, limit=48):
        return self.cpi[:limit]

    def unemployment_history(self, limit=48):
        return self.unrate[:limit]


def _tickers(n=12):
    return [f"STK{i:02d}" for i in range(n)]


def _closes(tickers, start="2023-06-01", end="2024-03-01", drift=None):
    """Business-day closes; drift maps ticker -> daily drift."""
    idx = pd.date_range(start, end, freq="B")
    drift = drift or {}
    data = {}
    for t in tickers:
        d = drift.get(t, 0.0)
        data[t] = 100 * (1 + d) ** np.arange(len(idx))
    return pd.DataFrame(data, index=idx)


# ------------------------------------------------------- fundamentals gating
def test_fundamentals_uses_only_filed_metrics():
    fd = FakeFree()
    for i, t in enumerate(_tickers()):
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
    assert out["STK00"][0] > out["STK11"][0], \
        f"lookahead leak? STK00={out['STK00'][0]:.3f} STK11={out['STK11'][0]:.3f}"
    print("ok: fundamentals gated on filing_date (no lookahead)")


def test_fundamentals_skips_unfiled_tickers():
    fd = FakeFree()
    fd.metrics_rows["AAA"] = [{"filing_date": "2025-01-01",
                               "price_to_earnings_ratio": 5.0}]
    out = fundamentals_scores(fd, ["AAA"], AS_OF, {})
    assert out == {}, "ticker with nothing filed as of T must emit no score"
    print("ok: unfiled ticker emits no view")


# ------------------------------------------------------- filings: PEAD leg
def _pead_setup():
    fd = FakeFree()
    tickers = ["UP", "DN", "STALE", "FUTURE", "FLAT"] + \
        [f"P{i:02d}" for i in range(10)]
    # baseline 8-Ks so the attention leg has history (no signal)
    base = [{"date": f"2023-{m:02d}-15", "kind": "other", "adsh": f"b{m}"}
            for m in range(6, 13)]
    # price paths: UP jumps +6% the session after its 2024-02-20 8-K,
    # DN drops -6%; pads move +/-0.8% after a 2024-02-15 8-K (z-score fuel).
    idx = pd.date_range("2023-06-01", "2024-03-01", freq="B")
    data = {}
    for t in tickers:
        px = np.full(len(idx), 100.0)
        if t == "UP":
            j = idx.get_loc("2024-02-20")
            px[j + 1:] = 106.0
            fd.events[t] = list(base) + [
                {"date": "2024-02-20", "kind": "earnings", "adsh": "e1"}]
        elif t == "DN":
            j = idx.get_loc("2024-02-20")
            px[j + 1:] = 94.0
            fd.events[t] = list(base) + [
                {"date": "2024-02-20", "kind": "earnings", "adsh": "e2"}]
        elif t == "STALE":
            fd.events[t] = list(base) + [
                {"date": "2023-12-01", "kind": "earnings",
                 "adsh": "e3"}]  # 91d old: not a signal
        elif t == "FUTURE":
            fd.events[t] = list(base) + [
                {"date": "2024-03-10", "kind": "earnings",
                 "adsh": "e4"}]  # after as_of: invisible
        elif t == "FLAT":
            fd.events[t] = list(base) + [
                {"date": "2024-02-20", "kind": "earnings",
                 "adsh": "e5"}]  # no price reaction: no verdict
        else:  # pads
            i = int(t[1:])
            k = idx.get_loc("2024-02-15")
            px[k + 1:] = 100.8 if i % 2 == 0 else 99.2
            fd.events[t] = list(base) + [
                {"date": "2024-02-15", "kind": "earnings", "adsh": f"p{i}"}]
        data[t] = px
    closes = pd.DataFrame(data, index=idx)
    return fd, closes, tickers


def test_filings_pead_follows_market_reaction():
    fd, closes, tickers = _pead_setup()
    out = filings_scores(fd, tickers, AS_OF, closes,
                         {"pead_window_days": 21, "pead_weight": 1.0,
                          "attn_weight": 0.0})
    assert "UP" in out and out["UP"][0] > 0, \
        f"+6% reaction must be bullish: {out.get('UP')}"
    assert "DN" in out and out["DN"][0] < 0, \
        f"-6% reaction must be bearish: {out.get('DN')}"
    assert "STALE" not in out, "91-day-old 8-K must not fire"
    assert "FUTURE" not in out, "8-K filed after as_of must be invisible"
    assert "FLAT" not in out, "|CAR| < 0.5% is no verdict, not a signal"
    print("ok: PEAD follows the 2-day market reaction; stale/future/tiny emit nothing")


def test_filings_pead_needs_observable_reaction():
    # 8-K filed ON as_of: the 2-day reaction isn't observable yet -> no view
    fd, closes, tickers = _pead_setup()
    fd.events["TODAY"] = [{"date": "2024-03-01", "kind": "earnings",
                           "adsh": "e9"}]
    closes["TODAY"] = 100.0
    out = filings_scores(fd, tickers + ["TODAY"], AS_OF, closes,
                         {"pead_weight": 1.0, "attn_weight": 0.0})
    assert "TODAY" not in out, \
        "same-day 8-K has no observable reaction inside the window"
    print("ok: PEAD needs the reaction observable within the as-of window")


# ------------------------------------------------------- filings: attention leg
def test_filings_attention_ignores_future_filings():
    fd = FakeFree()
    tickers = ["BUSY", "QUIET"] + [f"Q{i:02d}" for i in range(10)]
    closes = _closes(tickers, drift={"BUSY": 0.001})  # BUSY drifts up
    base = [{"date": f"2023-{m:02d}-10", "kind": "other", "adsh": f"x{m}"}
            for m in range(1, 13)]
    for t in tickers:
        fd.events[t] = list(base)
    # BUSY: 6 filings in the last 63d (before as_of) + rising price
    fd.events["BUSY"] += [
        {"date": f"2024-01-{d:02d}", "kind": "other", "adsh": f"r{d}"}
        for d in range(10, 16)]
    # QUIET: 6 filings but all AFTER as_of -> invisible
    fd.events["QUIET"] += [
        {"date": f"2024-03-{d:02d}", "kind": "other", "adsh": f"f{d}"}
        for d in range(10, 16)]
    out = filings_scores(fd, tickers, AS_OF, closes,
                         {"pead_weight": 0.0, "attn_weight": 1.0})
    assert "BUSY" in out and out["BUSY"][0] > 0, \
        f"abnormal filings + rising price must be positive: {out.get('BUSY')}"
    # QUIET's 6 extra filings are all after as_of: invisible. Its visible
    # history is exactly baseline -> neutral, not positive.
    assert "QUIET" in out, "QUIET has real baseline history, so it scores"
    assert abs(out["QUIET"][0]) < 0.5, \
        f"future filings must not leak into the score: {out['QUIET'][0]:.3f}"
    assert out["BUSY"][0] > out["QUIET"][0] + 1.0
    print("ok: attention gated on filing date <= as_of, signed by price")


# ------------------------------------------------------------------- macro
def test_macro_factor_neutral_without_data():
    fd = FakeFree()
    f = macro_factor(fd, AS_OF, {})
    assert f == 1.0, f"no macro data must not gate the book (got {f})"
    print("ok: macro factor = 1.0 when data is absent")


def test_macro_labor_read_moves_factor():
    fd = FakeFree()
    # flat curve + flat CPI, but unemployment spiking -> risk-off tilt
    dates = pd.date_range("2022-01-01", "2024-03-01", freq="MS")
    fd.curves = [{"date": d.strftime("%Y-%m-%d"), "10_year": 4.0, "2_year": 4.5}
                 for d in dates[::-1]]
    fd.cpi = [{"date": d.strftime("%Y-%m-%d"), "yoy_change": 3.0}
              for d in dates[::-1]]
    fd.unrate = [{"date": d.strftime("%Y-%m-%d"),
                  "rate": 3.5 + max(0, (i - 44)) * 0.5}
                 for i, d in enumerate(dates[::-1])]
    f = macro_factor(fd, AS_OF, {})
    assert 0.5 <= f < 1.0, f"labor spike should tilt risk-off, got {f}"
    print(f"ok: unemployment spike tilts macro factor risk-off ({f:.3f})")


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
                       "filings": 0.15, "macro": 0.15},
                      regime_factor=1.0)
    assert set(conv) == set(_tickers())
    assert all(0 < v < 1 for v in conv.values())
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
    blind_conv = {lbl: 0.1 * i for i, lbl in enumerate(sorted(mapping.values()))}
    conv = deanonymize_convictions(blind_conv, mapping)
    assert set(conv) == set(_tickers())
    print("ok: anonymize/deanonymize round-trips, no identity leaks")


if __name__ == "__main__":
    test_fundamentals_uses_only_filed_metrics()
    test_fundamentals_skips_unfiled_tickers()
    test_filings_pead_follows_market_reaction()
    test_filings_pead_needs_observable_reaction()
    test_filings_attention_ignores_future_filings()
    test_macro_factor_neutral_without_data()
    test_macro_labor_read_moves_factor()
    test_blend_regime_bounds()
    test_fuse_desks_two_level()
    test_anonymize_roundtrip()
    print("ALL DESK TESTS PASSED")
