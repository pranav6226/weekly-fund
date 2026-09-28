"""Unit tests for the Phase 5 paper-trading loop.

Covers: dry-run makes zero HTTP calls, the 3% per-position cap, paper
endpoint enforcement, and the pre-submit sanity guards.

Run:  ./.venv/bin/python -m pytest tests/   (or plain python tests/test_paper_trade.py)
"""
from __future__ import annotations

import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig
from fund.decision.portfolio import construct_targets
from fund.execution.broker import (
    PAPER_API_BASE,
    BrokerError,
    DryRunBroker,
    PaperBroker,
)
from fund.live.paper import (
    MAX_ORDER_PCT_OF_EQUITY,
    PlannedOrder,
    apply_position_cap,
    build_order_plan,
    check_order_sanity,
    most_recent_friday,
)

TODAY = date(2026, 9, 25)  # a Friday


def paper_cfg(**kw):
    base = dict(max_positions=20, max_position_weight=0.03,
                max_sector_weight=0.30, entry_rank=20, exit_rank=30,
                stop_loss=0.10)
    base.update(kw)
    return FundConfig(**base)


# ------------------------------------------------------------------ broker
def test_paper_endpoint_is_the_only_allowed():
    assert PAPER_API_BASE == "https://paper-api.alpaca.markets"
    try:
        PaperBroker(api_key="k", api_secret="s",
                    base_url="https://api.alpaca.markets")
    except ValueError:
        pass
    else:
        raise AssertionError("live endpoint must be refused")
    try:
        PaperBroker(api_key="k", api_secret="s",
                    base_url="https://evil.example.com")
    except ValueError:
        pass
    else:
        raise AssertionError("arbitrary endpoint must be refused")
    print("ok: non-paper endpoints refused")


def test_dry_run_broker_makes_no_http_calls():
    real_urlopen = urllib.request.urlopen

    def _boom(*a, **k):
        raise AssertionError("network call attempted in dry-run")

    urllib.request.urlopen = _boom
    try:
        b = DryRunBroker(
            positions={"AAPL": {"qty": 10, "market_value": 2000.0,
                                "avg_entry_price": 190.0, "current_price": 200.0}},
            account_equity=100_000.0)
        assert b.get_account()["equity"] == 100_000.0
        assert b.get_positions()["AAPL"]["qty"] == 10
        # planning + sanity on the dry-run view: still no network
        plan = build_order_plan({"AAPL": 0.03, "MSFT": 0.02},
                                b.get_positions(), 100_000.0,
                                {"AAPL": 200.0, "MSFT": 300.0})
        check_order_sanity(plan, 100_000.0, b.get_positions())
        try:
            b.submit_order("AAPL", "buy", notional=100.0)
        except BrokerError:
            pass
        else:
            raise AssertionError("DryRunBroker must never submit")
    finally:
        urllib.request.urlopen = real_urlopen
    print("ok: dry-run path makes zero HTTP calls")


# ------------------------------------------------------------------ planning
def test_per_position_cap_respected():
    conv = {f"T{i}": 0.9 - i * 0.01 for i in range(60)}
    prices = {f"T{i}": 100.0 for i in range(60)}
    sectors = {f"T{i}": f"S{i % 5}" for i in range(60)}
    t = construct_targets(conv, {}, prices, sectors, paper_cfg(), as_of=TODAY,
                          regime_factor=1.0)
    t = apply_position_cap(t, 0.03)  # live loop clips the conviction tilt
    assert t, "should hold something"
    assert max(t.values()) <= 0.03 + 1e-9, f"cap breach: {max(t.values())}"
    print(f"ok: {len(t)} targets, max weight {max(t.values()):.3%} <= 3%")


def test_order_plan_buys_and_sells():
    targets = {"AAPL": 0.03, "MSFT": 0.02}
    positions = {
        "AAPL": {"qty": 5, "market_value": 1000.0, "avg_entry_price": 190.0,
                 "current_price": 200.0},
        "TSLA": {"qty": 10, "market_value": 2500.0, "avg_entry_price": 240.0,
                 "current_price": 250.0},
    }
    plan = build_order_plan(targets, positions, 100_000.0,
                            {"AAPL": 200.0, "MSFT": 300.0, "TSLA": 250.0})
    by_sym = {o.symbol: o for o in plan}
    assert by_sym["AAPL"].side == "buy"
    assert abs(by_sym["AAPL"].notional - 2000.0) < 1e-6  # 3% of 100k - 1k held
    assert by_sym["MSFT"].side == "buy"
    assert abs(by_sym["MSFT"].notional - 2000.0) < 1e-6
    assert by_sym["TSLA"].side == "sell"  # held but not a target -> exit
    assert abs(by_sym["TSLA"].qty - 10) < 1e-9
    check_order_sanity(plan, 100_000.0, positions)
    print("ok: plan buys to target, exits non-targets")


def test_sell_never_exceeds_held_qty():
    # stale low price must not turn a trim into a short
    targets = {"AAPL": 0.005}
    positions = {"AAPL": {"qty": 10, "market_value": 2000.0,
                          "avg_entry_price": 190.0, "current_price": 200.0}}
    plan = build_order_plan(targets, positions, 100_000.0, {"AAPL": 1.0})
    assert len(plan) == 1 and plan[0].side == "sell"
    assert plan[0].qty <= 10 + 1e-9, "sell capped at held qty"
    print("ok: sells capped at held quantity")


def test_sanity_rejects_oversize_order():
    big = [PlannedOrder("AAPL", "buy", notional=10_000.0, qty=None,
                        est_value=10_000.0, reason="x")]
    try:
        check_order_sanity(big, 100_000.0, {})
    except ValueError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("oversize order must be rejected")
    assert MAX_ORDER_PCT_OF_EQUITY == 0.032
    print("ok: >3.2% orders rejected")


def test_sanity_rejects_short():
    short = [PlannedOrder("AAPL", "sell", notional=None, qty=5.0,
                          est_value=1000.0, reason="x")]
    try:
        check_order_sanity(short, 100_000.0, {})  # nothing held
    except ValueError as e:
        assert "short" in str(e)
    else:
        raise AssertionError("short sale must be rejected")
    print("ok: shorts rejected")


def test_most_recent_friday():
    idx = pd.date_range("2026-09-21", "2026-09-28", freq="B")  # Mon..Mon
    assert most_recent_friday(idx, today=date(2026, 9, 28)) == date(2026, 9, 25)
    assert most_recent_friday(idx, today=date(2026, 9, 27)) == date(2026, 9, 25)
    print("ok: review date resolves to the most recent Friday")


if __name__ == "__main__":
    test_paper_endpoint_is_the_only_allowed()
    test_dry_run_broker_makes_no_http_calls()
    test_per_position_cap_respected()
    test_order_plan_buys_and_sells()
    test_sell_never_exceeds_held_qty()
    test_sanity_rejects_oversize_order()
    test_sanity_rejects_short()
    test_most_recent_friday()
    print("ALL PAPER-TRADE TESTS PASSED")
