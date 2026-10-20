"""Tests for the option-B rebalance deferral (scripts/autonomous_worker.py).

The collision: a Friday 13:35 PT stop-check exit fills at Monday's
open, so it is still open at the Saturday 09:00 PT rebalance slot and
the job-start open-orders guard aborted the whole week's rebalance
(it happened Sat 2026-10-17, INTC pending). Option B: open orders at
the Saturday slot itself DEFER the rebalance to the Monday 07:00 PT
slot instead of aborting; everything else still aborts.

The scenario dates below ARE that weekend: SAT = 2026-10-17 09:00 PT
(review Friday 2026-10-16), MON = 2026-10-19 07:00 PT. All subprocess
and broker seams are stubbed -- no network, no broker, no orders.

Run: ./.venv/bin/python tests/test_rebalance_deferral.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import autonomous_worker as worker  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
SAT = datetime(2026, 10, 17, 9, 0, tzinfo=PT)   # review = Fri 2026-10-16
MON = datetime(2026, 10, 19, 7, 0, tzinfo=PT)
REVIEW = "2026-10-16"


class _Env:
    """Save/restore the worker's env switches around a test."""

    KEYS = ("FUND_AUTONOMOUS_LIVE", "FUND_TRADING_HALTED", "FUND_STRICT")

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Stubs:
    """Replace the worker's subprocess/broker/alert seams with recorders.

    open_orders is a plain attribute: a test sets it to whatever the
    account shows at that point in the simulated weekend.
    """

    def __init__(self, tmp_path, open_orders=0, trade_rc=0):
        self.calls: list[tuple[str, list]] = []
        self.alerts: list[str] = []
        self.open_orders = open_orders
        self._trade_rc = trade_rc
        self._tmp = tmp_path

    def __enter__(self):
        self._orig = (worker.run_cmd, worker.open_orders_count,
                      worker.maybe_alert, worker.RUNS_DIR)

        def fake_run_cmd(name, cmd, timeout):
            self.calls.append((name, list(cmd)))
            if name == "rebalance":
                return self._trade_rc, "trade output"
            return 0, ""

        worker.run_cmd = fake_run_cmd
        worker.open_orders_count = lambda: self.open_orders
        worker.maybe_alert = lambda text: self.alerts.append(text)
        worker.RUNS_DIR = self._tmp / "paper_runs"
        return self

    def __exit__(self, *exc):
        (worker.run_cmd, worker.open_orders_count,
         worker.maybe_alert, worker.RUNS_DIR) = self._orig

    def marker(self, name: str) -> Path:
        return self._tmp / "paper_runs" / name

    def trade_calls(self) -> list[list]:
        return [cmd for name, cmd in self.calls if name == "rebalance"]


def _live():
    os.environ["FUND_AUTONOMOUS_LIVE"] = "1"


# ------------------------------------------------------- (1) Saturday defer
def test_saturday_open_orders_defers_and_submits_nothing(tmp_path):
    # The Oct 17 scenario: one INTC stop exit pending at the slot.
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)
        assert st.calls == [], "deferral: no refresh, no trade step"
        assert not st.marker(f"autonomous-{REVIEW}.json").exists(), \
            "deferral must not write the Saturday marker"
        pending = state.get("deferred_rebalance")
        assert pending is not None, state
        assert pending["review_date"] == REVIEW
        assert pending["saturday"] == "2026-10-17"
        assert pending["open_orders"] == 1
        assert any("deferred" in a for a in st.alerts), st.alerts
    print("ok: Saturday with a pending stop exit defers, submits nothing")


# ------------------------------------------------- (2) normal Saturday path
def test_saturday_no_open_orders_normal_path_unchanged(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=0) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)
        assert [n for n, _ in st.calls] == ["panel-refresh", "rebalance"]
        trade_cmd = st.trade_calls()[0]
        assert "--as-of" in trade_cmd and REVIEW in trade_cmd
        assert "--live" in trade_cmd and "--strict" in trade_cmd
        assert st.marker(f"autonomous-{REVIEW}.json").exists()
        assert "deferred_rebalance" not in state
        assert not st.marker(f"autonomous-deferred-{REVIEW}.json").exists()
    print("ok: Saturday with no open orders runs the normal path as before")


# ------------------------------------------- (3) Monday deferred run executes
def test_monday_deferred_run_executes_after_fills(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)          # Saturday: defer
        assert st.calls == []
        st.open_orders = 0                        # Monday open: INTC filled
        worker.job_deferred_rebalance(MON, state)
        assert [n for n, _ in st.calls] == ["panel-refresh", "rebalance"]
        trade_cmd = st.trade_calls()[0]
        # Still Friday's panel/targets input (--as-of the review date);
        # the diff happens inside paper_trade against the live account
        # as it stands on Monday -- no Saturday order plan is replayed.
        assert "--as-of" in trade_cmd and REVIEW in trade_cmd
        assert "--live" in trade_cmd and "--strict" in trade_cmd
        assert st.marker(f"autonomous-deferred-{REVIEW}.json").exists()
        assert "deferred_rebalance" not in state, "deferral consumed"
    print("ok: Monday deferred run executes the rebalance after the fills")


# ------------------------------ (4) Monday orders still open: abort, consumed
def test_monday_deferred_run_orders_still_open_aborts(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)
        st.open_orders = 2                        # still open on Monday
        worker.job_deferred_rebalance(MON, state)
        assert st.calls == [], "still open on Monday: nothing may run"
        assert not st.marker(f"autonomous-deferred-{REVIEW}.json").exists()
        assert "deferred_rebalance" not in state, \
            "deferral consumed -- no second deferral, no loop"
        assert any("deferred rebalance" in a and "aborted" in a
                   for a in st.alerts), st.alerts
        # And it stays dead: a later Monday call is a no-op.
        st.open_orders = 0
        worker.job_deferred_rebalance(MON, state)
        assert st.calls == []
    print("ok: orders still open on Monday abort the week, deferral consumed")


# ------------------------------------- (5) halts on Saturday abort, no defer
def test_kill_switch_on_saturday_aborts_not_defers(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        os.environ["FUND_TRADING_HALTED"] = "1"
        state: dict = {}
        worker.job_rebalance(SAT, state)
        assert st.calls == []
        assert "deferred_rebalance" not in state
        assert not st.marker(f"autonomous-{REVIEW}.json").exists()
    print("ok: kill switch on Saturday refuses -- never defers")


def test_drawdown_halt_on_saturday_aborts_not_defers(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.update_drawdown(state, 100_000.0)
        worker.update_drawdown(state, 90_000.0)  # -10%: persisted halt
        assert state.get("trading_halted") is True
        worker.job_rebalance(SAT, state)
        assert st.calls == []
        assert "deferred_rebalance" not in state
    print("ok: persisted drawdown halt on Saturday refuses -- never defers")


# --------------------------- (6) strict degraded data on Saturday: no defer
def test_strict_degraded_saturday_aborts_not_defers(tmp_path):
    # No open orders, panel fine, but paper_trade --strict aborts on
    # degraded decision inputs (trade step exit 1). The week ends as
    # an abort: marker written, failure alerted, NO deferral recorded.
    with _Env(), _Stubs(tmp_path, open_orders=0, trade_rc=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)
        assert [n for n, _ in st.calls] == ["panel-refresh", "rebalance"]
        assert st.marker(f"autonomous-{REVIEW}.json").exists()
        assert "deferred_rebalance" not in state
        assert any("trade step failed" in a for a in st.alerts), st.alerts
        # Nothing leaks into Monday either.
        worker.job_deferred_rebalance(MON, state)
        assert len(st.trade_calls()) == 1
    print("ok: strict degraded-data abort on Saturday never becomes a deferral")


# --------------------------------------- (7) no double-run of the same slot
def test_no_double_run_of_same_saturday_slot(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)          # defers
        first = dict(state["deferred_rebalance"])
        worker.job_rebalance(SAT, state)          # same slot again
        assert state["deferred_rebalance"] == first, \
            "a repeat Saturday gate hit must not rewrite the deferral"
        assert st.calls == []
        st.open_orders = 0
        worker.job_deferred_rebalance(MON, state)  # executes exactly once
        assert len(st.trade_calls()) == 1
        worker.job_deferred_rebalance(MON, state)  # consumed: no-op
        assert len(st.trade_calls()) == 1
        # Even a stray Saturday call now skips: the deferred marker
        # for this review date exists.
        worker.job_rebalance(SAT, state)
        assert len(st.trade_calls()) == 1
    print("ok: the same Saturday slot can never run twice")


# ------------------------------------------- extra guards around the design
def test_open_orders_check_failure_never_defers(tmp_path):
    # -1 = the check itself failed: not explicable as weekend fills.
    with _Env(), _Stubs(tmp_path, open_orders=-1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)
        assert st.calls == []
        assert "deferred_rebalance" not in state
        assert any("aborted" in a for a in st.alerts), st.alerts
    print("ok: a failed open-orders check aborts -- never defers")


def test_monday_without_deferral_is_noop(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=0) as st:
        _live()
        state: dict = {}
        worker.job_deferred_rebalance(MON, state)
        assert st.calls == [] and state == {}
    print("ok: a Monday with no pending deferral does nothing")


def test_deferred_run_halted_on_monday_aborts_and_consumes(tmp_path):
    with _Env(), _Stubs(tmp_path, open_orders=1) as st:
        _live()
        state: dict = {}
        worker.job_rebalance(SAT, state)          # defers on Saturday
        assert "deferred_rebalance" in state
        st.open_orders = 0
        os.environ["FUND_TRADING_HALTED"] = "1"   # halt lands Sunday
        worker.job_deferred_rebalance(MON, state)
        assert st.calls == [], "halted Monday: nothing may run"
        assert "deferred_rebalance" not in state, "deferral consumed"
    print("ok: a halt appearing by Monday aborts the deferred run, consumed")


if __name__ == "__main__":
    import tempfile

    test_saturday_open_orders_defers_and_submits_nothing(
        Path(tempfile.mkdtemp()))
    test_saturday_no_open_orders_normal_path_unchanged(
        Path(tempfile.mkdtemp()))
    test_monday_deferred_run_executes_after_fills(Path(tempfile.mkdtemp()))
    test_monday_deferred_run_orders_still_open_aborts(
        Path(tempfile.mkdtemp()))
    test_kill_switch_on_saturday_aborts_not_defers(Path(tempfile.mkdtemp()))
    test_drawdown_halt_on_saturday_aborts_not_defers(
        Path(tempfile.mkdtemp()))
    test_strict_degraded_saturday_aborts_not_defers(Path(tempfile.mkdtemp()))
    test_no_double_run_of_same_saturday_slot(Path(tempfile.mkdtemp()))
    test_open_orders_check_failure_never_defers(Path(tempfile.mkdtemp()))
    test_monday_without_deferral_is_noop(Path(tempfile.mkdtemp()))
    test_deferred_run_halted_on_monday_aborts_and_consumes(
        Path(tempfile.mkdtemp()))
    print("ALL REBALANCE-DEFERRAL TESTS PASSED")
