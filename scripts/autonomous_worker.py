#!/usr/bin/env python3
"""Autonomous scheduler for the weekly fund (Railway worker).

Replaces the human-launched weekly rhythm with a self-driving loop,
all times America/Los_Angeles:

  daily 17:05   scripts/monitor.py          (read-only account check)
  Mon   07:00   scripts/monitor.py again    (fill check after the open)
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
    (e.g. a manual run already queued this week's trades).
  * FUND_AUTONOMOUS_LIVE must be exactly "1" for the Saturday job to
    submit orders; anything else makes it a dry-run.
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
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PT = ZoneInfo("America/Los_Angeles")
STATE_PATH = ROOT / "results" / "worker_state.json"
RUNS_DIR = ROOT / "results" / "paper_runs"

MONITOR_TIME = (17, 5)     # daily
MONDAY_CHECK_TIME = (7, 0)  # Mondays
REBALANCE_TIME = (9, 0)    # Saturdays
SATURDAY = 5               # datetime.weekday(): Mon=0 .. Sun=6
MONDAY = 0


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


def job_monitor(label: str) -> None:
    rc, out = run_cmd(label, [PY, "scripts/monitor.py"], timeout=900)
    if "ANOMALY" in out.upper():
        maybe_alert(f"weekly-fund monitor ANOMALY ({label}):\n{out[-1200:]}")


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


def job_rebalance(instant: datetime) -> None:
    review = (instant - timedelta(days=1)).date()  # the Friday of this run
    marker = RUNS_DIR / f"autonomous-{review}.json"
    if marker.exists():
        log(f"rebalance {review}: marker exists, skipping (already ran)")
        return
    n_open = open_orders_count()
    if n_open != 0:
        log(f"rebalance {review}: ABORT -- open_orders={n_open} "
            "(expected 0; a manual run may have queued orders, or the "
            "check failed). No orders submitted by the worker.")
        maybe_alert(f"weekly-fund rebalance {review} aborted: "
                    f"open_orders={n_open}")
        return
    live = os.environ.get("FUND_AUTONOMOUS_LIVE", "") == "1"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "review_date": str(review), "started_at": datetime.now(PT).isoformat(),
        "mode": "live" if live else "dry-run",
    }, indent=2))
    rc, _ = run_cmd("panel-refresh", [PY, "scripts/download_history.py"],
                    timeout=10800)
    if rc != 0:
        log(f"rebalance {review}: panel refresh failed (exit {rc}); "
            "skipping trade step -- never trade on a stale panel")
        maybe_alert(f"weekly-fund {review}: panel refresh failed, "
                    "rebalance skipped")
        return
    cmd = [PY, "scripts/paper_trade.py", "--as-of", str(review)]
    if live:
        cmd.append("--live")
    rc, _ = run_cmd("rebalance", cmd, timeout=3600)
    log(f"rebalance {review}: finished exit {rc} "
        f"({'live' if live else 'dry-run'})")


# ------------------------------------------------------------------ loop

JOBS = [
    # (name, last-instant fn, runner, is_trade_job)
    ("monitor_daily",
     lambda now: last_daily(now, *MONITOR_TIME),
     lambda: job_monitor("monitor_daily"), False),
    ("monitor_monday",
     lambda now: last_weekly(now, MONDAY, *MONDAY_CHECK_TIME),
     lambda: job_monitor("monitor_monday"), False),
    ("rebalance_weekly",
     lambda now: last_weekly(now, SATURDAY, *REBALANCE_TIME),
     None, True),  # runner needs the instant; handled specially
]


def tick(state: dict) -> None:
    now = datetime.now(PT)
    done = state.setdefault("done", {})
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
        log(f"{name}: due (scheduled {key})")
        if is_trade:
            job_rebalance(instant)
        else:
            runner()
        done[name] = key
        save_state(state)


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
    while True:
        try:
            tick(state)
        except Exception as e:  # noqa: BLE001 - the loop must survive a bad tick
            log(f"tick error: {e!r}")
        import time
        time.sleep(60)


if __name__ == "__main__":
    main()
