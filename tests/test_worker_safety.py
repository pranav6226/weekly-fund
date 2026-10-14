"""Safety tests for scripts/autonomous_worker.py (audit M4b, M5, M6).

The worker previously had zero tests. These cover the decision logic:
alert triggers, the kill switch / drawdown halt gate on the trade step,
the -8% drawdown computation, the pre-trade open-orders re-check, and
that the live trade step is invoked with --strict. All subprocess and
broker seams are stubbed -- nothing here touches the network or a
broker.

Run: ./.venv/bin/python tests/test_worker_safety.py
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
SAT = datetime(2026, 10, 17, 9, 0, tzinfo=PT)  # review = Fri 2026-10-16


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
    """Replace the worker's subprocess/broker/alert seams with recorders."""

    def __init__(self, tmp_path, open_orders=0, trade_rc=0):
        self.calls: list[tuple[str, list]] = []
        self.alerts: list[str] = []
        self._open_orders = open_orders
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
        worker.open_orders_count = lambda: self._open_orders
        worker.maybe_alert = lambda text: self.alerts.append(text)
        worker.RUNS_DIR = self._tmp / "paper_runs"
        return self

    def __exit__(self, *exc):
        (worker.run_cmd, worker.open_orders_count,
         worker.maybe_alert, worker.RUNS_DIR) = self._orig


# --------------------------------------------------------------------- M5
def test_monitor_alert_decisions():
    assert worker.monitor_needs_alert(0, "STATUS: HEALTHY") is False
    assert worker.monitor_needs_alert(0, "STATUS: ANOMALY") is True
    assert worker.monitor_needs_alert(2, "ERROR: broker down") is True
    assert worker.monitor_needs_alert(0, "ERROR: weird but exit 0") is True
    assert worker.monitor_needs_alert(-1, "") is True  # timeout
    print("ok: monitor alerts on anomaly, error text, and nonzero exit")


def test_alerts_expected_only_when_live():
    with _Env():
        assert worker.alerts_expected() is False
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        assert worker.alerts_expected() is True
    print("ok: alerts expected iff FUND_AUTONOMOUS_LIVE=1")


def test_trade_step_failure_alerts(tmp_path):
    with _Env(), _Stubs(tmp_path, trade_rc=3) as st:
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        worker.job_rebalance(SAT, {})
        assert [n for n, _ in st.calls] == ["panel-refresh", "rebalance"]
        assert any("trade step failed" in a for a in st.alerts), st.alerts
    print("ok: nonzero trade-step exit fires an alert")


def test_live_trade_step_is_strict(tmp_path):
    with _Env(), _Stubs(tmp_path) as st:
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        worker.job_rebalance(SAT, {})
        trade_cmd = dict(st.calls)["rebalance"]
        assert "--live" in trade_cmd and "--strict" in trade_cmd, trade_cmd
    print("ok: live trade step is invoked with --live --strict")


# --------------------------------------------------------------------- M4b
def test_open_orders_recheck_blocks_trade(tmp_path):
    # Job-start check passes (0 open), then orders appear during the
    # panel refresh: the re-check must block the trade step entirely.
    with _Env(), _Stubs(tmp_path) as st:
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        seq = iter([0, 2])
        worker.open_orders_count = lambda: next(seq)
        worker.job_rebalance(SAT, {})
        names = [n for n, _ in st.calls]
        assert names == ["panel-refresh"], names  # trade step never ran
        assert any("pre-trade re-check" in a for a in st.alerts), st.alerts
    print("ok: open orders appearing mid-refresh block the trade step")


# --------------------------------------------------------------------- M6
def test_drawdown_trips_at_8_not_5():
    state: dict = {}
    assert worker.update_drawdown(state, 100_000.0) == 0.0
    dd = worker.update_drawdown(state, 95_000.0)
    assert abs(dd - 0.05) < 1e-9 and not state.get("trading_halted")
    dd = worker.update_drawdown(state, 91_000.0)
    assert abs(dd - 0.09) < 1e-9 and state.get("trading_halted") is True
    assert "drawdown" in worker.halt_reason(state)
    # peak never ratchets down; the halt never auto-clears on a bounce
    worker.update_drawdown(state, 99_000.0)
    assert state["peak_equity"] == 100_000.0
    assert state.get("trading_halted") is True
    print("ok: drawdown breaker trips at -9%, not at -5%, never auto-clears")


def test_halt_flag_blocks_trade_step(tmp_path):
    state: dict = {}
    worker.update_drawdown(state, 100_000.0)
    worker.update_drawdown(state, 90_000.0)  # -10%: halted
    assert state.get("trading_halted") is True
    with _Env(), _Stubs(tmp_path) as st:
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        worker.job_rebalance(SAT, state)
        worker.job_rebalance(SAT, state)  # second refusal: no re-alert
        assert st.calls == [], "halted: no panel refresh, no trade step"
        refusals = [a for a in st.alerts if "REFUSED" in a]
        assert len(refusals) == 1, st.alerts  # one alert per review date
        marker = st._tmp / "paper_runs" / "autonomous-2026-10-16.json"
        assert not marker.exists(), "halted run must not write the marker"
    print("ok: persisted halt flag blocks the trade step (alert once)")


def test_kill_switch_blocks_trade_step(tmp_path):
    with _Env(), _Stubs(tmp_path) as st:
        os.environ["FUND_AUTONOMOUS_LIVE"] = "1"
        os.environ["FUND_TRADING_HALTED"] = "1"
        assert "kill switch" in worker.halt_reason({})
        worker.job_rebalance(SAT, {})
        assert st.calls == [], "kill switch: nothing may run"
    print("ok: FUND_TRADING_HALTED=1 blocks the worker trade step")


def test_monitor_job_tracks_drawdown_and_alerts(tmp_path):
    outputs = iter([
        (0, "account ACTIVE | equity $100,000.00 | buying power $1.00\n"
            "STATUS: HEALTHY"),
        (0, "account ACTIVE | equity $91,500.00 | buying power $1.00\n"
            "STATUS: HEALTHY"),
    ])
    alerts: list[str] = []
    orig_run, orig_alert = worker.run_cmd, worker.maybe_alert
    worker.run_cmd = lambda name, cmd, timeout: next(outputs)
    worker.maybe_alert = lambda text: alerts.append(text)
    try:
        state: dict = {}
        assert worker.job_monitor("monitor_daily", state) is True
        assert state["peak_equity"] == 100_000.0
        assert not state.get("trading_halted")
        assert worker.job_monitor("monitor_daily", state) is True
        assert state.get("trading_halted") is True  # -8.5% from peak
        assert any("DRAWDOWN BREAKER" in a for a in alerts), alerts
    finally:
        worker.run_cmd, worker.maybe_alert = orig_run, orig_alert
    print("ok: monitor job tracks peak equity and trips the breaker")


def test_monitor_job_alerts_on_crash():
    orig_run, orig_alert = worker.run_cmd, worker.maybe_alert
    alerts: list[str] = []
    worker.run_cmd = lambda name, cmd, timeout: (2, "ERROR: alpaca down")
    worker.maybe_alert = lambda text: alerts.append(text)
    try:
        assert worker.job_monitor("monitor_daily", {}) is False
        assert alerts, "a crashed monitor must alert"
    finally:
        worker.run_cmd, worker.maybe_alert = orig_run, orig_alert
    print("ok: monitor job alerts when the monitor itself crashes")


def test_parse_equity():
    assert worker.parse_equity("equity $101,274.45 | x") == 101274.45
    assert worker.parse_equity("equity $900.10") == 900.10
    assert worker.parse_equity("no numbers here") is None
    assert worker.parse_equity("") is None
    print("ok: equity parsed from monitor summary; absent -> None")


if __name__ == "__main__":
    import tempfile

    test_monitor_alert_decisions()
    test_alerts_expected_only_when_live()
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_trade_step_failure_alerts(tp / "a")
        test_live_trade_step_is_strict(tp / "b")
        test_open_orders_recheck_blocks_trade(tp / "c")
        test_halt_flag_blocks_trade_step(tp / "d")
        test_kill_switch_blocks_trade_step(tp / "e")
        test_monitor_job_tracks_drawdown_and_alerts(tp / "f")
    test_drawdown_trips_at_8_not_5()
    test_monitor_job_alerts_on_crash()
    test_parse_equity()
    print("ALL WORKER SAFETY TESTS PASSED")
