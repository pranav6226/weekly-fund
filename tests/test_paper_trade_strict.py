"""Strict mode + live-path guards for scripts/paper_trade.py.

Audit findings covered:
  H2  strict mode: a Jev failure (total or partial) aborts the run --
      no fusion fallback, no backfill -- before anything is planned.
  M4a the live path itself refuses when the account has open orders.
  M6a FUND_TRADING_HALTED=1 kill switch.
  M7b panel-Friday staleness date logic.

Run: ./.venv/bin/python tests/test_paper_trade_strict.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_trade  # noqa: E402
from fund.decision.jev import JevError, JevPartialError  # noqa: E402
from fund.live.paper import expected_recent_friday, panel_friday_is_stale  # noqa: E402
from fund.research.schema import AgentScore  # noqa: E402

AS_OF = date(2026, 10, 9)  # a Friday
REGIME = SimpleNamespace(label="risk-on", factor=1.0)
WEIGHTS = {"technicals": 0.4, "fundamentals": 0.3,
           "filings": 0.15, "macro": 0.15}


def _scores():
    out = []
    for t, z in (("AAA", 1.0), ("BBB", -0.5)):
        out.append(AgentScore(t, "trend", z, 0.9, "trend", AS_OF))
        out.append(AgentScore(t, "fundamentals", z, 0.8, "fund", AS_OF))
    return out


class FakeJev:
    """JevClient stand-in: convictions() raises or returns per `mode`."""

    def __init__(self, mode):
        self.mode = mode
        self.last_stats = {"model": "fake-1.0", "n_tickers": 2,
                           "n_requests": 1, "n_cached": 0,
                           "mean_confidence": 0.5}

    def convictions(self, scores, **kw):
        if self.mode == "error":
            raise JevError("jev is down")
        if self.mode == "partial":
            raise JevPartialError("proxy ate a batch", {"AAA": 0.9})
        return {"AAA": 0.9, "BBB": 0.2}


def _with_fake_jev(mode, fn):
    orig_client = paper_trade.JevClient
    orig_key = os.environ.get("JEV_API_KEY")
    paper_trade.JevClient = lambda: FakeJev(mode)
    os.environ["JEV_API_KEY"] = "test-key"
    try:
        fn()
    finally:
        paper_trade.JevClient = orig_client
        if orig_key is None:
            os.environ.pop("JEV_API_KEY", None)
        else:
            os.environ["JEV_API_KEY"] = orig_key


def _decide(mode, strict, tmp):
    return paper_trade.decide_convictions(
        _scores(), WEIGHTS, 1.0, REGIME, tmp / "jev.json",
        strict=strict, use_jev=True)


def _expect_abort(fn, needle):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"abort message should say {needle!r}: {e}"
        return
    raise AssertionError("strict mode must abort (SystemExit)")


def test_strict_jev_error_aborts(tmp_path):
    _with_fake_jev("error", lambda: _expect_abort(
        lambda: _decide("error", True, tmp_path), "STRICT"))
    print("ok: strict + total Jev failure aborts, no fusion fallback")


def test_strict_jev_partial_aborts(tmp_path):
    _with_fake_jev("partial", lambda: _expect_abort(
        lambda: _decide("partial", True, tmp_path), "STRICT"))
    print("ok: strict + partial Jev failure aborts, no backfill")


def test_nonstrict_jev_error_falls_back(tmp_path):
    def run():
        conv, source = _decide("error", False, tmp_path)
        assert source == "fusion"
        assert set(conv) == {"AAA", "BBB"}  # fusion judged both tickers

    _with_fake_jev("error", run)
    print("ok: non-strict + Jev failure keeps the fusion fallback")


def test_nonstrict_jev_partial_backfills(tmp_path):
    def run():
        conv, source = _decide("partial", False, tmp_path)
        assert source == "jev (partial) + fusion backfill"
        assert conv["AAA"] == 0.9  # Jev's judgment kept
        assert "BBB" in conv  # missing ticker backfilled by fusion

    _with_fake_jev("partial", run)
    print("ok: non-strict + partial Jev backfills only the missing tickers")


def test_strict_jev_success_unchanged(tmp_path):
    def run():
        conv, source = _decide("ok", True, tmp_path)
        assert conv == {"AAA": 0.9, "BBB": 0.2}
        assert source == "jev fake-1.0"

    _with_fake_jev("ok", run)
    print("ok: strict + healthy Jev proceeds exactly as before")


# ------------------------------------------------------------------ M4a
class _FakeBroker:
    def __init__(self, open_orders):
        self._open = open_orders
        self.submitted = []

    def get_open_orders(self):
        return self._open

    def submit_order(self, *a, **k):
        self.submitted.append((a, k))
        return {"id": "x", "status": "accepted"}


def test_open_orders_guard_aborts():
    broker = _FakeBroker([{"id": "o1", "symbol": "AAA", "status": "new"}])
    try:
        paper_trade.refuse_if_open_orders(broker)
    except SystemExit as e:
        assert "open order" in str(e)
    else:
        raise AssertionError("open orders must abort the live path")
    assert broker.submitted == [], "nothing may be submitted"
    paper_trade.refuse_if_open_orders(_FakeBroker([]))  # none open: passes
    print("ok: live path refuses with open orders, proceeds with none")


# ------------------------------------------------------------------ M6a
def test_kill_switch_env():
    orig = os.environ.get("FUND_TRADING_HALTED")
    try:
        os.environ.pop("FUND_TRADING_HALTED", None)
        assert paper_trade.kill_switch_active() is False
        os.environ["FUND_TRADING_HALTED"] = "1"
        assert paper_trade.kill_switch_active() is True
        os.environ["FUND_TRADING_HALTED"] = "yes"  # only exact "1" counts
        assert paper_trade.kill_switch_active() is False
    finally:
        if orig is None:
            os.environ.pop("FUND_TRADING_HALTED", None)
        else:
            os.environ["FUND_TRADING_HALTED"] = orig
    print("ok: kill switch reads FUND_TRADING_HALTED=1 exactly")


# ------------------------------------------------------------------ M7b
def test_expected_recent_friday():
    assert expected_recent_friday(date(2026, 10, 16)) == date(2026, 10, 16)  # Fri
    assert expected_recent_friday(date(2026, 10, 17)) == date(2026, 10, 16)  # Sat
    assert expected_recent_friday(date(2026, 10, 18)) == date(2026, 10, 16)  # Sun
    assert expected_recent_friday(date(2026, 10, 19)) == date(2026, 10, 16)  # Mon
    assert expected_recent_friday(date(2026, 10, 14)) == date(2026, 10, 9)   # Wed
    print("ok: expected Friday = most recent Friday on/before today")


def test_panel_friday_staleness():
    # Wed Oct 14: a panel through Fri Oct 9 is as fresh as it can be.
    assert panel_friday_is_stale(date(2026, 10, 9), date(2026, 10, 14)) is False
    # Fri Oct 16: the panel should have reached today; Oct 9 is stale.
    assert panel_friday_is_stale(date(2026, 10, 9), date(2026, 10, 16)) is True
    assert panel_friday_is_stale(date(2026, 10, 16), date(2026, 10, 16)) is False
    print("ok: staleness = panel Friday older than the expected Friday")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_strict_jev_error_aborts(tp)
        test_strict_jev_partial_aborts(tp)
        test_nonstrict_jev_error_falls_back(tp)
        test_nonstrict_jev_partial_backfills(tp)
        test_strict_jev_success_unchanged(tp)
    test_open_orders_guard_aborts()
    test_kill_switch_env()
    test_expected_recent_friday()
    test_panel_friday_staleness()
    print("ALL PAPER-TRADE STRICT TESTS PASSED")
