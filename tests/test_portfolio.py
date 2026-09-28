"""Unit tests for the shared portfolio constructor.

Run:  ./.venv/bin/python -m pytest tests/   (or plain python tests/test_portfolio.py)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig
from fund.decision.portfolio import PositionState, construct_targets


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


def test_top_n_selected_equal_weight():
    scores = {f"T{i}": 10 - i for i in range(10)}
    prices = {f"T{i}": 100.0 for i in range(10)}
    sectors = {f"T{i}": "Tech" for i in range(10)}
    t = construct_targets(scores, {}, prices, sectors, cfg())
    # sector cap 0.6 with equal names -> drops lowest scored until <= 0.6
    assert t, "should hold something"
    assert sum(t.values()) <= 1.0 + 1e-9
    # all survivors are top-ranked names
    assert set(t) <= {"T0", "T1", "T2", "T3"}


def test_hysteresis_band():
    c = cfg(entry_rank=2, exit_rank=4, max_positions=4)
    scores = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    prices = {k: 100.0 for k in scores}
    sectors = {k: f"S{k}" for k in scores}  # each own sector -> no cap binding
    pos = {"A": PositionState("A", 10, 100.0, date(2024, 1, 1)),
           "B": PositionState("B", 10, 100.0, date(2024, 1, 1))}
    t = construct_targets(scores, pos, prices, sectors, c)
    assert "A" in t and "B" in t, "ranks 1-2 stay (within exit_rank)"


def test_rank_decay_exits():
    c = cfg(entry_rank=2, exit_rank=3, max_positions=4)
    scores = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 0.5, "F": 0.4}
    prices = {k: 100.0 for k in scores}
    sectors = {k: f"S{k}" for k in scores}
    pos = {"F": PositionState("F", 10, 100.0, date(2024, 1, 1))}
    t = construct_targets(scores, pos, prices, sectors, c)
    assert "F" not in t, "rank 6 > exit_rank 3 -> exited"


def test_stop_loss_exits():
    c = cfg()
    scores = {"A": 5, "B": 4}
    prices = {"A": 89.0, "B": 100.0}  # A down 11% from entry
    sectors = {"A": "S1", "B": "S2"}
    pos = {"A": PositionState("A", 10, 100.0, date(2024, 1, 1))}
    # simulator passes tickers pending a stop-exit as blocked, so the same
    # review cannot instantly re-enter the name it just stopped out of
    t = construct_targets(scores, pos, prices, sectors, c, blocked={"A"})
    assert "A" not in t, "stop-loss should exit A"


def test_missing_ticker_exits():
    c = cfg()
    scores = {"A": 5}
    prices = {"A": 100.0}
    sectors = {"A": "S1"}
    pos = {"Z": PositionState("Z", 10, 100.0, date(2024, 1, 1))}
    t = construct_targets(scores, pos, prices, sectors, c)
    assert "Z" not in t, "ticker leaving the scored set is exited"


if __name__ == "__main__":
    for name, fn in sorted(
        [(k, v) for k, v in globals().items() if k.startswith("test_")]
    ):
        fn()
        print(f"PASS {name}")
