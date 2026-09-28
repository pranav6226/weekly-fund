"""Unit tests for the shared portfolio constructor (v2: convictions,
time-stop, conviction tilt, regime scaling).

Run:  ./.venv/bin/python -m pytest tests/   (or plain python tests/test_portfolio.py)
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig
from fund.decision.fusion import fuse_scores, fuse_scores_regime, market_regime
from fund.decision.portfolio import PositionState, construct_targets
from fund.research.schema import AgentScore

TODAY = date(2024, 3, 1)


def cfg(**kw):
    base = dict(
        max_positions=4,
        max_position_weight=0.5,
        max_sector_weight=0.6,
        entry_rank=4,
        exit_rank=6,
        stop_loss=0.10,
    )
    base.update(kw)
    return FundConfig(**base)


def call(conv, pos, prices, sectors, c, **kw):
    kw.setdefault("as_of", TODAY)
    return construct_targets(conv, pos, prices, sectors, c, **kw)


def test_top_n_selected_equal_weight():
    conv = {f"T{i}": 0.99 - i * 0.01 for i in range(10)}
    prices = {f"T{i}": 100.0 for i in range(10)}
    sectors = {f"T{i}": "Tech" for i in range(10)}
    t = call(conv, {}, prices, sectors, cfg())
    assert t, "should hold something"
    assert sum(t.values()) <= 1.0 + 1e-9
    assert set(t) <= {"T0", "T1", "T2", "T3"}


def test_hysteresis_band():
    c = cfg(entry_rank=2, exit_rank=4, max_positions=4)
    conv = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5, "F": 0.4}
    prices = {k: 100.0 for k in conv}
    sectors = {k: f"S{k}" for k in conv}
    pos = {"A": PositionState("A", 10, 100.0, date(2024, 1, 15)),
           "B": PositionState("B", 10, 100.0, date(2024, 1, 15))}
    t = call(conv, pos, prices, sectors, c)
    assert "A" in t and "B" in t, "ranks 1-2 stay (within exit_rank)"


def test_rank_decay_exits():
    c = cfg(entry_rank=2, exit_rank=3, max_positions=4)
    conv = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5, "F": 0.4}
    prices = {k: 100.0 for k in conv}
    sectors = {k: f"S{k}" for k in conv}
    pos = {"F": PositionState("F", 10, 100.0, date(2024, 1, 15))}
    t = call(conv, pos, prices, sectors, c)
    assert "F" not in t, "rank 6 > exit_rank 3 -> exited"


def test_stop_loss_exits():
    c = cfg()
    conv = {"A": 0.9, "B": 0.8}
    prices = {"A": 89.0, "B": 100.0}  # A down 11% from entry
    sectors = {"A": "S1", "B": "S2"}
    pos = {"A": PositionState("A", 10, 100.0, date(2024, 1, 15))}
    t = call(conv, pos, prices, sectors, c, blocked={"A"})
    assert "A" not in t, "stop-loss should exit A"


def test_missing_ticker_exits():
    c = cfg()
    conv = {"A": 0.9}
    prices = {"A": 100.0}
    sectors = {"A": "S1"}
    pos = {"Z": PositionState("Z", 10, 100.0, date(2024, 1, 15))}
    t = call(conv, pos, prices, sectors, c)
    assert "Z" not in t, "ticker leaving the scored set is exited"


def test_time_stop_exits_stale_position():
    c = cfg(max_hold_days=84)
    conv = {"A": 0.9, "B": 0.8}
    prices = {"A": 100.0, "B": 100.0}
    sectors = {"A": "S1", "B": "S2"}
    stale = {"A": PositionState("A", 10, 100.0, TODAY - timedelta(days=100))}
    fresh = {"A": PositionState("A", 10, 100.0, TODAY - timedelta(days=10))}
    t_stale = call(conv, stale, prices, sectors, c)
    t_fresh = call(conv, fresh, prices, sectors, c)
    assert "A" not in t_stale, "held 100d > 84d time-stop -> exited"
    assert "A" in t_fresh, "held 10d stays"


def test_regime_factor_scales_to_cash():
    c = cfg()
    conv = {"A": 0.9, "B": 0.8}
    prices = {"A": 100.0, "B": 100.0}
    sectors = {"A": "S1", "B": "S2"}
    full = call(conv, {}, prices, sectors, c, regime_factor=1.0)
    half = call(conv, {}, prices, sectors, c, regime_factor=0.5)
    assert abs(sum(full.values()) - 2 * sum(half.values())) < 1e-9
    assert sum(half.values()) < 0.6, "remainder stays as cash"


def test_conviction_tilt_redistributes():
    c = cfg(max_positions=2, max_position_weight=1.0, max_sector_weight=1.0)
    conv = {"A": 0.99, "B": 0.51}  # A much stronger than B
    prices = {"A": 100.0, "B": 100.0}
    sectors = {"A": "S1", "B": "S2"}
    t = call(conv, {}, prices, sectors, c)
    assert t["A"] > t["B"], "higher conviction gets more weight"
    assert abs(sum(t.values()) - 1.0) < 1e-9, "tilt redistributes, does not lever"


def test_fuse_scores_calibrated():
    c = cfg()
    scores = [
        AgentScore("A", "trend", 2.0, 0.5, "", TODAY),
        AgentScore("A", "reversal", -1.0, 0.5, "", TODAY),
        AgentScore("B", "trend", -2.0, 0.5, "", TODAY),
    ]
    out = fuse_scores(scores, c)
    assert 0.0 < out["B"] < 0.5 < out["A"] < 1.0
    # weights: trend .35 reversal .15 -> A: (2*.35 + -1*.15)/.5 = 1.1 -> Phi ~0.86
    assert 0.8 < out["A"] < 0.95, f"A conviction {out['A']}"


def test_market_regime_risk_off():
    import numpy as np
    import pandas as pd
    c = cfg()
    # synthetic crash: steady then -40% with high vol
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    px = np.concatenate([np.linspace(100, 120, 300), np.linspace(120, 70, 100)])
    closes = pd.DataFrame({f"T{i}": px * (1 + 0.001 * i) for i in range(20)}, index=idx)
    r = market_regime(closes, idx[-1], c)
    assert r.factor < 1.0, f"crash should de-risk, got {r}"
    assert r.factor >= c.regime_floor


def test_regime_fusion_rotates_weights():
    c = cfg()
    scores = [
        AgentScore("A", "trend", 2.0, 0.5, "", TODAY),     # strong trend
        AgentScore("A", "reversal", -2.0, 0.5, "", TODAY),  # weak reversal
        AgentScore("B", "trend", -2.0, 0.5, "", TODAY),
        AgentScore("B", "reversal", 2.0, 0.5, "", TODAY),
    ]
    on = fuse_scores_regime(scores, c, 1.0)
    off = fuse_scores_regime(scores, c, 0.5)
    # risk-on: trend dominates -> A wins; risk-off: reversal dominates -> B wins
    assert on["A"] > on["B"], f"risk-on should favor trend: {on}"
    assert off["B"] > off["A"], f"risk-off should favor reversal: {off}"
    # smooth interpolation, no bucket jumps
    mid = fuse_scores_regime(scores, c, 0.75)
    assert on["A"] > mid["A"] > off["A"]


if __name__ == "__main__":
    for name, fn in sorted(
        [(k, v) for k, v in globals().items() if k.startswith("test_")]
    ):
        fn()
        print(f"PASS {name}")
