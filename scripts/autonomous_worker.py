#!/usr/bin/env python3
"""Autonomous scheduler for the weekly fund (Railway worker).

Replaces the human-launched weekly rhythm with a self-driving loop,
all times America/Los_Angeles:

  daily 17:05   scripts/monitor.py          (read-only account check)
  Mon   07:00   scripts/monitor.py again    (fill check after the open)
                -> a deferred Saturday rebalance, if one is pending
                (see the option-B deferral note under Safety)
  Mon-Thu 13:35 scripts/stop_check.py       (daily stop-loss exits;
                --live under the same guards as the rebalance --
                added 2026-10-16 to match the validated backtest,
                whose simulator enforces stops on daily closes;
                Fridays excluded by design -- see STOP_DAYS)
  Sat   09:00   scripts/download_history.py (panel refresh)
                -> scripts/paper_trade.py --live

Safety properties:
  * The rebalance NEVER fires for a scheduled instant that predates
    this worker's first boot -- no catch-up trading after a deploy or
    a volume reset.
  * A marker file for the review date is written BEFORE the live run
    starts, so a crash/restart mid-run cannot double-submit the same
    rebalance on the next boot.
  * The rebalance aborts if the paper account has ANY open orders
    (e.g. a manual run already queued this week's trades) -- checked
    at job start AND re-checked immediately before the trade step,
    after the (up to 3h) panel refresh.
  * ONE exception to the job-start abort (option B, approved
    2026-10-20): open orders at the Saturday slot itself DEFER the
    rebalance instead of killing the week. A Friday stop-check exit
    fills at Monday's open, so it is still open on Saturday morning
    through no fault of the rebalance. The worker records the
    deferral in the persisted state and the trade step runs at the
    Monday 07:00 slot, re-diffed against the post-fill account but
    still sized on Friday's panel (--as-of the review date). If
    orders are still open on Monday, the week is aborted entirely --
    no second deferral. Every other refusal (kill switch, drawdown
    halt, degraded data, a FAILED open-orders check, orders appearing
    mid-refresh) aborts exactly as before and never defers.
  * FUND_AUTONOMOUS_LIVE must be exactly "1" for the Saturday job to
    submit orders; anything else makes it a dry-run.
  * The live trade step always runs paper_trade.py --strict: any
    degraded decision input (desk failure, missing macro reads, Jev
    failure) aborts instead of silently falling back.
  * Kill switch: FUND_TRADING_HALTED=1 makes the trade step refuse
    (paper_trade.py enforces it for manual runs too).
  * Drawdown breaker: the daily monitor tracks peak equity in the
    persisted state; at >=8% below peak it sets a persisted halt flag
    and the trade step refuses until an operator resets the state.
  * Execution stays locked to the Alpaca PAPER endpoint by PaperBroker
    itself; this worker adds no new order path.
  * If the panel refresh fails, the live run is skipped -- never trade
    on a stale panel.

State lives in results/worker_state.json (results/ is on the Railway
volume via scripts/worker_entry.sh), so restarts keep the schedule.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PT = ZoneInfo("America/Los_Angeles")
STATE_PATH = ROOT / "results" / "worker_state.json"
RUNS_DIR = ROOT / "results" / "paper_runs"

MONITOR_TIME = (17, 5)     # daily
MONDAY_CHECK_TIME = (7, 0)  # Mondays
STOP_TIME = (13, 35)       # Mon-Thu, after the 13:00 PT close
REBALANCE_TIME = (9, 0)    # Saturdays
SATURDAY = 5               # datetime.weekday(): Mon=0 .. Sun=6
MONDAY = 0
WEEKDAYS = {0, 1, 2, 3, 4}
# The stop check deliberately skips Friday: a Friday-close breach is
# caught by Saturday's review (construct_targets applies the same
# stop) and fills at the same Monday open, while a Friday stop ORDER
# would sit open all weekend and trip the rebalance's open-order
# guard -- the collision that blocked the 2026-10-17 rebalance.
STOP_DAYS = {0, 1, 2, 3}  # Mon-Thu


def log(msg: str) -> None:
    print(f"[worker {datetime.now(PT):%Y-%m-%d %H:%M:%S} PT] {msg}", flush=True)


# ------------------------------------------------------------------ state

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001 - missing/corrupt state = fresh start
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


# ------------------------------------------------------------------ schedule

def last_daily(now: datetime, hh: int, mm: int) -> datetime:
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return cand if cand <= now else cand - timedelta(days=1)


def last_weekly(now: datetime, weekday: int, hh: int, mm: int) -> datetime:
    cand = last_daily(now, hh, mm)
    while cand.weekday() != weekday:
        cand -= timedelta(days=1)
    return cand


def last_weekday(now: datetime, hh: int, mm: int,
                 days: set = WEEKDAYS) -> datetime:
    """Most recent instant at hh:mm on one of `days` (Mon=0)."""
    cand = last_daily(now, hh, mm)
    while cand.weekday() not in days:
        cand -= timedelta(days=1)
    return cand


# ------------------------------------------------------------------ jobs

def run_cmd(name: str, cmd: list[str], timeout: int) -> tuple[int, str]:
    log(f"{name}: running {' '.join(cmd[1:])}")
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        tail = "\n".join(out.strip().splitlines()[-15:])
        log(f"{name}: exit {p.returncode}\n{tail}")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        log(f"{name}: TIMEOUT after {timeout}s")
        return -1, ""


def maybe_alert(text: str) -> None:
    url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"text": text[-1800:]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=15)  # noqa: S310 - operator-set URL
    except Exception as e:  # noqa: BLE001 - alerting is best-effort
        log(f"alert webhook failed: {e}")


# ------------------------------------------------------- halts & alerts

DRAWDOWN_HALT_PCT = 0.08  # equity >=8% below its peak halts trading

_EQUITY_RE = re.compile(r"equity \$([0-9][0-9,]*\.[0-9]{2})")


def alerts_expected() -> bool:
    """Alerts are expected whenever unattended live trading is on."""
    return os.environ.get("FUND_AUTONOMOUS_LIVE", "") == "1"


def monitor_needs_alert(rc: int, out: str) -> bool:
    """Alert on anomalies AND on a crashed/errored monitor itself.

    Previously only the string "ANOMALY" alerted, so a monitor that
    died printing "ERROR" (exit 2) failed silently -- the fund could
    look 'quiet' for weeks while nothing was actually being checked.
    """
    if rc != 0:
        return True
    up = (out or "").upper()
    return "ANOMALY" in up or "ERROR" in up


def parse_equity(text: str) -> float | None:
    """Equity from monitor.py's summary line ('equity $101,274.45')."""
    m = _EQUITY_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def update_drawdown(state: dict, equity: float) -> float:
    """Track peak equity in the persisted state; trip the halt at -8%.

    Returns the current drawdown fraction (0.0 = at peak). Once
    tripped, state['trading_halted'] STAYS set: it clears only when an
    operator edits/resets the persisted state file
    (results/worker_state.json). It never auto-clears -- a bounce off
    the lows must not silently re-arm trading.
    """
    peak = max(float(state.get("peak_equity") or 0.0), equity)
    state["peak_equity"] = peak
    dd = (peak - equity) / peak if peak > 0 else 0.0
    if dd >= DRAWDOWN_HALT_PCT and not state.get("trading_halted"):
        state["trading_halted"] = True
        state["halt_reason"] = (
            f"drawdown breaker: equity ${equity:,.2f} is {dd:.1%} below "
            f"peak ${peak:,.2f} (>= {DRAWDOWN_HALT_PCT:.0%})")
        state["halted_at"] = datetime.now(PT).isoformat()
    return dd


def halt_reason(state: dict) -> str | None:
    """Why trading is halted, or None. Two independent switches: the
    FUND_TRADING_HALTED=1 env kill switch, and the persisted drawdown
    halt flag (clears only by resetting the persisted state)."""
    if os.environ.get("FUND_TRADING_HALTED", "") == "1":
        return "kill switch FUND_TRADING_HALTED=1"
    if state.get("trading_halted"):
        return str(state.get("halt_reason")
                   or "drawdown breaker (persisted halt flag set)")
    return None


def job_monitor(label: str, state: dict) -> bool:
    rc, out = run_cmd(label, [PY, "scripts/monitor.py"], timeout=900)
    if monitor_needs_alert(rc, out):
        maybe_alert(f"weekly-fund monitor ({label}) rc={rc}:\n{out[-1200:]}")
    equity = parse_equity(out)
    if equity is not None:
        tripped_before = bool(state.get("trading_halted"))
        dd = update_drawdown(state, equity)
        if state.get("trading_halted") and not tripped_before:
            log(f"DRAWDOWN BREAKER TRIPPED: {state.get('halt_reason')}")
            maybe_alert(
                f"weekly-fund DRAWDOWN BREAKER: {state.get('halt_reason')} "
                "-- trading is HALTED until an operator resets the "
                "persisted worker state")
        elif dd > 0:
            log(f"drawdown tracker: equity ${equity:,.2f} vs peak "
                f"${state['peak_equity']:,.2f} ({dd:.1%} below)")
    return rc == 0


def open_orders_count() -> int:
    """Open orders on the paper account; -1 if the check itself failed."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from fund.execution.broker import PaperBroker
        orders = PaperBroker()._request(
            "GET", "/v2/orders?status=open&limit=500")
        return len(orders or [])
    except Exception as e:  # noqa: BLE001
        log(f"open-orders check failed: {e}")
        return -1


def job_rebalance(instant: datetime, state: dict) -> None:
    review = (instant - timedelta(days=1)).date()  # the Friday of this run
    reason = halt_reason(state)
    if reason is not None:
        # Halted (kill switch or drawdown breaker): refuse BEFORE the
        # marker is written, and alert at most once per review date.
        # Note the week's slot is still consumed by the scheduler --
        # after clearing a halt, run that week's rebalance manually.
        log(f"rebalance {review}: REFUSED -- trading halted ({reason}). "
            "No marker written, no orders submitted.")
        alerts = state.setdefault("halt_alerts", {})
        if alerts.get(str(review)) != reason:
            maybe_alert(f"weekly-fund rebalance {review} REFUSED: "
                        f"trading halted ({reason})")
            alerts[str(review)] = reason
        return
    marker = RUNS_DIR / f"autonomous-{review}.json"
    deferred_marker = RUNS_DIR / f"autonomous-deferred-{review}.json"
    if marker.exists() or deferred_marker.exists():
        log(f"rebalance {review}: marker exists, skipping (already ran)")
        return
    n_open = open_orders_count()
    if n_open == -1:
        # The check ITSELF failed: not explicable as pending weekend
        # fills, so it aborts exactly as before -- never deferred.
        log(f"rebalance {review}: ABORT -- open-orders check failed "
            "(expected 0 open orders). No orders submitted by the "
            "worker.")
        maybe_alert(f"weekly-fund rebalance {review} aborted: "
                    "open-orders check failed")
        return
    if n_open > 0:
        # Option B: orders open at the Saturday slot itself -- the
        # classic case is a Friday stop-check exit that fills at
        # Monday's open -- DEFER the rebalance to Monday 07:00 PT
        # instead of aborting the week. Orders appearing LATER (the
        # pre-trade re-check below) still abort: those are not the
        # explicable weekend-fill case.
        _record_deferral(instant, review, n_open, state)
        return
    _execute_rebalance(review, deferred=False)


def _record_deferral(instant: datetime, review: date, n_open: int,
                     state: dict) -> None:
    """Persist a Saturday -> Monday rebalance deferral (option B).

    Only the first pending deferral is kept: a second Saturday gate
    hit while one is outstanding (should be unreachable -- the Monday
    job consumes it before the next Saturday) never overwrites it.
    No marker is written: the marker means 'a trade run started', and
    the deferred run writes its own on Monday.
    """
    if "deferred_rebalance" in state:
        log(f"rebalance {review}: a deferral is already pending for "
            f"{state['deferred_rebalance'].get('review_date')}; "
            "keeping it and skipping this slot without trading")
        return
    state["deferred_rebalance"] = {
        "review_date": str(review),
        "saturday": instant.date().isoformat(),
        "deferred_at": datetime.now(PT).isoformat(),
        "open_orders": n_open,
        "reason": "open orders at the Saturday slot "
                  "(pending weekend fills)",
    }
    log(f"rebalance {review}: DEFERRED to Monday 07:00 PT -- "
        f"open_orders={n_open} at the Saturday slot (a Friday "
        "stop-check exit fills at Monday's open). No marker written, "
        "no orders submitted.")
    maybe_alert(f"weekly-fund rebalance {review} deferred to Monday "
                f"07:00 PT: open_orders={n_open}")


def _execute_rebalance(review: date, *, deferred: bool) -> None:
    """Marker -> panel refresh -> pre-trade re-check -> trade step.

    Shared by the normal Saturday path and the Monday deferred run.
    The deferred run gets its own marker file so the marker-before-run
    discipline (a crash/restart cannot double-submit) holds
    independently for each path; the normal marker content is
    unchanged. The trade step always runs paper_trade.py --as-of the
    review (panel) Friday, so a Monday run diffs Friday's targets
    against the live account as it stands at run time -- no Saturday
    order plan is ever cached or replayed.
    """
    live = os.environ.get("FUND_AUTONOMOUS_LIVE", "") == "1"
    marker = RUNS_DIR / (f"autonomous-deferred-{review}.json" if deferred
                         else f"autonomous-{review}.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "review_date": str(review), "started_at": datetime.now(PT).isoformat(),
        "mode": "live" if live else "dry-run",
        **({"deferred": True} if deferred else {}),
    }, indent=2))
    rc, _ = run_cmd("panel-refresh", [PY, "scripts/download_history.py"],
                    timeout=10800)
    if rc != 0:
        log(f"rebalance {review}: panel refresh failed (exit {rc}); "
            "skipping trade step -- never trade on a stale panel")
        maybe_alert(f"weekly-fund {review}: panel refresh failed, "
                    "rebalance skipped")
        return
    # Re-check open orders IMMEDIATELY before the trade step: the panel
    # refresh above can run for hours, and a manual run in that window
    # may have queued orders the job-start check could not see.
    n_open = open_orders_count()
    if n_open != 0:
        log(f"rebalance {review}: ABORT -- open_orders={n_open} on the "
            "pre-trade re-check (orders appeared during the panel "
            "refresh). No orders submitted by the worker.")
        maybe_alert(f"weekly-fund rebalance {review} aborted on the "
                    f"pre-trade re-check: open_orders={n_open}")
        return
    cmd = [PY, "scripts/paper_trade.py", "--as-of", str(review)]
    if live:
        cmd += ["--live", "--strict"]
    rc, _ = run_cmd("rebalance", cmd, timeout=3600)
    if rc != 0:
        maybe_alert(f"weekly-fund rebalance {review}: trade step failed "
                    f"(exit {rc}); no retry -- see the worker log")
    log(f"rebalance {review}: finished exit {rc} "
        f"({'live' if live else 'dry-run'})")


def job_deferred_rebalance(instant: datetime, state: dict) -> None:
    """Monday 07:00 PT: run a Saturday rebalance deferred by option B.

    Runs right after the Monday monitor in the same tick, i.e. after
    the 06:30 PT open fills the weekend's pending orders. The
    deferral is consumed FIRST (popped from the persisted state),
    whatever happens next: a halt, still-open orders, or a failed
    trade step aborts that week's rebalance entirely -- there is no
    second deferral and no retry loop.
    """
    pending = state.pop("deferred_rebalance", None)
    if not pending:
        return
    review = date.fromisoformat(str(pending["review_date"]))
    log(f"deferred rebalance {review}: consuming the deferral recorded "
        f"{pending.get('deferred_at')} (Saturday slot "
        f"{pending.get('saturday')}, open_orders="
        f"{pending.get('open_orders')} at deferral)")
    reason = halt_reason(state)
    if reason is not None:
        log(f"deferred rebalance {review}: ABORT -- trading halted "
            f"({reason}). Deferral consumed; the week is skipped.")
        alerts = state.setdefault("halt_alerts", {})
        if alerts.get(str(review)) != reason:
            maybe_alert(f"weekly-fund deferred rebalance {review} "
                        f"ABORTED: trading halted ({reason})")
            alerts[str(review)] = reason
        return
    marker = RUNS_DIR / f"autonomous-deferred-{review}.json"
    if marker.exists():
        log(f"deferred rebalance {review}: marker exists, skipping "
            "(already ran)")
        return
    n_open = open_orders_count()
    if n_open != 0:
        log(f"deferred rebalance {review}: ABORT -- open_orders="
            f"{n_open} still open after Monday's open. Deferral "
            "consumed; no second deferral, no orders submitted.")
        maybe_alert(f"weekly-fund deferred rebalance {review} "
                    f"aborted: open_orders={n_open} on Monday")
        return
    _execute_rebalance(review, deferred=True)


def job_stop_check(instant: datetime, state: dict) -> None:
    """Weekday stop-loss enforcement (exits only), after the close.

    Same guard shape as job_rebalance: halt refusal BEFORE the slot
    marker, marker before the run, abort on any open orders, marker
    consumes the slot so the job never auto-retries. The script
    re-checks the kill switch and open orders itself before
    submitting, and only ever sells.
    """
    trade_date = instant.date()
    reason = halt_reason(state)
    if reason is not None:
        log(f"stop check {trade_date}: REFUSED -- trading halted "
            f"({reason}). No marker written, no orders submitted.")
        alerts = state.setdefault("halt_alerts", {})
        key = f"stop:{trade_date}"
        if alerts.get(key) != reason:
            maybe_alert(f"weekly-fund stop check {trade_date} REFUSED: "
                        f"trading halted ({reason})")
            alerts[key] = reason
        return
    marker = RUNS_DIR / f"autonomous-stop-{trade_date}.json"
    if marker.exists():
        log(f"stop check {trade_date}: marker exists, skipping "
            "(already ran)")
        return
    n_open = open_orders_count()
    if n_open != 0:
        log(f"stop check {trade_date}: ABORT -- open_orders={n_open} "
            "(expected 0; an earlier run may have queued orders, or "
            "the check failed). No orders submitted by the worker.")
        maybe_alert(f"weekly-fund stop check {trade_date} aborted: "
                    f"open_orders={n_open}")
        return
    live = os.environ.get("FUND_AUTONOMOUS_LIVE", "") == "1"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "trade_date": str(trade_date),
        "started_at": datetime.now(PT).isoformat(),
        "mode": "live" if live else "dry-run",
    }, indent=2))
    cmd = [PY, "scripts/stop_check.py", "--as-of", str(trade_date)]
    if live:
        cmd += ["--live"]
    rc, _ = run_cmd("stop_check", cmd, timeout=1200)
    if rc != 0:
        maybe_alert(f"weekly-fund stop check {trade_date}: failed "
                    f"(exit {rc}); no retry -- see the worker log")
    log(f"stop check {trade_date}: finished exit {rc} "
        f"({'live' if live else 'dry-run'})")


# ------------------------------------------------------------------ loop

JOBS = [
    # (name, last-instant fn, runner(state), is_trade_job)
    ("monitor_daily",
     lambda now: last_daily(now, *MONITOR_TIME),
     lambda state: job_monitor("monitor_daily", state), False),
    ("monitor_monday",
     lambda now: last_weekly(now, MONDAY, *MONDAY_CHECK_TIME),
     lambda state: job_monitor("monitor_monday", state), False),
    ("rebalance_deferred",
     lambda now: last_weekly(now, MONDAY, *MONDAY_CHECK_TIME),
     None, True),  # after monitor_monday in this list = same tick,
    # after the fill check; runner needs the instant, handled specially
    ("stop_check_daily",
     lambda now: last_weekday(now, *STOP_TIME, days=STOP_DAYS),
     None, True),  # runner needs the instant; handled specially
    ("rebalance_weekly",
     lambda now: last_weekly(now, SATURDAY, *REBALANCE_TIME),
     None, True),  # runner needs the instant; handled specially
]


def tick(state: dict) -> None:
    now = datetime.now(PT)
    done = state.setdefault("done", {})
    attempts = state.setdefault("attempts", {})
    boot = state["bootstrapped_at"]
    for name, instant_fn, runner, is_trade in JOBS:
        instant = instant_fn(now)
        key = instant.isoformat()
        if done.get(name, "") >= key:
            continue
        if is_trade and instant < datetime.fromisoformat(boot):
            # Deploy-time catch-up must never trade: mark and move on.
            log(f"{name}: skipping pre-boot instant {key} (no catch-up)")
            done[name] = key
            continue
        att = attempts.get(name, {})
        if att.get("instant") == key and not att.get("ok"):
            # A failed monitor retries, but at most every 15 minutes --
            # and the trade job is NEVER auto-retried (it marks done
            # below unconditionally; its own guards decide outcomes).
            if now - datetime.fromisoformat(att["at"]) < timedelta(minutes=15):
                continue
        log(f"{name}: due (scheduled {key})")
        if is_trade:
            if name == "rebalance_weekly":
                job_rebalance(instant, state)
            elif name == "rebalance_deferred":
                job_deferred_rebalance(instant, state)
            else:
                job_stop_check(instant, state)
            done[name] = key
        else:
            ok = bool(runner(state))
            attempts[name] = {"instant": key, "at": now.isoformat(), "ok": ok}
            if ok:
                done[name] = key
        save_state(state)


PROBE_HOSTS = ["paper-api.alpaca.markets", "query1.finance.yahoo.com",
               "api.stlouisfed.org", "www.sec.gov"]


def probe_network() -> None:
    """Boot-time reachability check, logged once per start so deploy
    logs show which data/execution endpoints the container can reach
    and over which address family."""
    for host in PROBE_HOSTS:
        try:
            with socket.create_connection((host, 443), timeout=8) as s:
                log(f"probe {host}: OK via {s.getpeername()[0]}")
        except Exception as e:  # noqa: BLE001
            log(f"probe {host}: FAIL {e}")


def main() -> None:
    if "--selftest" in sys.argv:
        now = datetime.now(PT)
        for name, fn, _, _ in JOBS:
            print(f"{name}: last instant {fn(now)}")
        print(f"state path: {STATE_PATH}")
        print(f"autonomous live: "
              f"{os.environ.get('FUND_AUTONOMOUS_LIVE', '') == '1'}")
        return
    state = load_state()
    if "bootstrapped_at" not in state:
        state["bootstrapped_at"] = datetime.now(PT).isoformat()
        save_state(state)
        log(f"first boot; bootstrapped_at={state['bootstrapped_at']}")
    log("autonomous worker started "
        f"(live={'1' if os.environ.get('FUND_AUTONOMOUS_LIVE') == '1' else 'dry-run only'})")
    if alerts_expected() and not os.environ.get("ALERT_WEBHOOK_URL", "").strip():
        log("WARNING: live trading is enabled (alerts expected) but "
            "ALERT_WEBHOOK_URL is not set -- aborts, trade failures and "
            "the drawdown breaker will NOT be paged anywhere")
    probe_network()
    while True:
        try:
            tick(state)
        except Exception as e:  # noqa: BLE001 - the loop must survive a bad tick
            log(f"tick error: {e!r}")
        import time
        time.sleep(60)


if __name__ == "__main__":
    main()
