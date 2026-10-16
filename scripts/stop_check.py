#!/usr/bin/env python3
"""Daily stop-loss enforcement for the weekly fund (exits only).

WHY THIS EXISTS: the validated v3 backtest's simulator scans every
position against the mandate's 10% stop on EVERY daily close (exit at
the next open). The live loop historically evaluated stops only inside
the Saturday review (construct_targets), so the live fund ran a weaker
risk policy than the one that was backtested. The 2026-10-15 A/B
backtest (results/backtest_daily_stop/metrics.json) quantified the
gap on the full 2020-2026 window: daily stops +331% total / 24.3%
CAGR / -26.8% max DD vs weekly-only +291% / 22.5% / -30.7%. This
script closes the gap: it is the live counterpart of simulator step 3.

BEHAVIOR (mirrors the simulator exactly):
  * For each real position (market value >= $1; dust is ignored),
    compare the as-of session's CLOSE to the position's average entry
    price. Close <= entry * (1 - stop_loss) -> exit the FULL position.
  * Runs after the close and submits DAY market orders, which fill at
    the next session's open -- the same decision/fill split as the sim.
  * Exits only. It never buys, never adds, never re-enters. The
    Saturday rebalance remains the only job that opens positions.

SAFETY (same posture as paper_trade.py / the autonomous worker):
  * Submits only with --live; the worker adds --live only when
    FUND_AUTONOMOUS_LIVE=1. Default is a dry-run that writes a record.
  * FUND_TRADING_HALTED=1 refuses live runs (checked here as the last
    line of defense; the worker also checks its persisted drawdown
    halt flag before invoking this script).
  * Refuses to submit while ANY order is open on the account.
  * A name is exited only on a verified fresh close (a daily bar dated
    exactly the as-of session). A name without one is CHECK_FAILED --
    never judged on stale data. Other names are still processed (a
    quote failure for one ticker must not leave a verified breach
    unexited), but the run then exits 2 so the worker alerts.
  * If NO held name has a bar dated as-of, the session almost
    certainly did not happen (holiday): status no_session, exit 0.

Record: results/paper_runs/stoprun-<date>.json (the worker's slot
marker is a separate file, autonomous-stop-<date>.json).

Usage: python scripts/stop_check.py [--live] [--as-of YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        os.environ.setdefault(var, "http://198.19.0.1:3128")

from fund.config import FundConfig  # noqa: E402
from fund.execution.broker import PaperBroker  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
RUNS_DIR = ROOT / "results" / "paper_runs"
MIN_POSITION_VALUE = 1.0  # below this a position is dust (monitor agrees)


def kill_switch_active() -> bool:
    """Manual kill switch: FUND_TRADING_HALTED=1 refuses live runs."""
    return os.environ.get("FUND_TRADING_HALTED", "") == "1"


def evaluate_stops(positions: dict, closes: dict, stop_loss: float) -> list:
    """Pure decision core. Returns one check dict per real position.

    positions: broker shape {sym: {qty, market_value, avg_entry_price}}
    closes:    {sym: close} containing ONLY closes verified fresh for
               the as-of session (the fetch layer enforces the date).
    A position whose symbol is missing from `closes`, or whose entry
    price is unusable, comes back status="check_failed" -- it is never
    exited on unverified data.
    """
    checks = []
    for sym in sorted(positions):
        p = positions[sym]
        mv = float(p.get("market_value") or 0)
        qty = float(p.get("qty") or 0)
        if abs(mv) < MIN_POSITION_VALUE or qty <= 0:
            continue  # dust / not a long position: not this job's concern
        entry = float(p.get("avg_entry_price") or 0)
        close = closes.get(sym)
        chk = {"symbol": sym, "qty": qty, "market_value": round(mv, 2),
               "avg_entry_price": entry, "close": close,
               "stop_level": None, "ret_pct": None,
               "status": "ok", "triggered": False}
        if close is None:
            chk["status"] = "check_failed"
            chk["reason"] = "no verified close for the as-of session"
        elif entry <= 0:
            chk["status"] = "check_failed"
            chk["reason"] = "avg_entry_price unavailable"
        else:
            chk["stop_level"] = round(entry * (1 - stop_loss), 4)
            chk["ret_pct"] = round(close / entry - 1, 4)
            chk["triggered"] = close <= entry * (1 - stop_loss)
        checks.append(chk)
    return checks


def fetch_closes_window(symbols: list[str], as_of: date) -> tuple[dict, int]:
    """Verified closes for the as-of session via Yahoo daily bars,
    fetched over a trailing 10-day window ending after as_of.

    Returns ({symbol: close}, n_fresh): only bars dated exactly as_of
    count as fresh. Symbols are translated to Yahoo style (BRK/B ->
    BRK-B); broker symbols are kept as keys.
    """
    import pandas as pd
    import yfinance as yf

    yahoo = {s: s.replace(".", "-").replace("/", "-") for s in symbols}
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=10)).date()
    end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()
    df = yf.download(list(yahoo.values()), start=str(start), end=str(end),
                     progress=False, auto_adjust=False)
    closes: dict[str, float] = {}
    n_fresh = 0
    if df is None or len(df) == 0:
        return closes, 0
    close_df = df["Close"]
    if not hasattr(close_df, "columns"):  # single-ticker frame
        close_df = close_df.to_frame(name=list(yahoo.values())[0])
    target = pd.Timestamp(as_of)
    for sym, ysym in yahoo.items():
        if ysym not in close_df.columns:
            continue
        s = close_df[ysym].dropna()
        if len(s) and pd.Timestamp(s.index[-1]) == target:
            closes[sym] = float(s.iloc[-1])
            n_fresh += 1
    return closes, n_fresh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="submit exit orders (default: dry-run)")
    ap.add_argument("--as-of", default=None,
                    help="session date to evaluate (default: today PT)")
    args = ap.parse_args()
    as_of = (date.fromisoformat(args.as_of) if args.as_of
             else datetime.now(PT).date())
    cfg = FundConfig()
    mode = "live" if args.live else "dry-run"
    print(f"stop check {as_of} -- mode={mode} stop_loss={cfg.stop_loss:.0%}",
          flush=True)

    if args.live and kill_switch_active():
        print("REFUSED: kill switch FUND_TRADING_HALTED=1", flush=True)
        return 2

    broker = PaperBroker()
    account = broker.get_account()
    if str(account.get("status")) != "ACTIVE":
        print(f"ABORT: account status {account.get('status')!r}", flush=True)
        return 2
    equity = float(account.get("equity") or 0)
    positions = broker.get_positions()
    real = {s: p for s, p in positions.items()
            if abs(float(p.get("market_value") or 0)) >= MIN_POSITION_VALUE
            and float(p.get("qty") or 0) > 0}
    print(f"account ACTIVE equity ${equity:,.2f}; {len(real)} real positions",
          flush=True)
    if not real:
        print("no real positions; nothing to check", flush=True)
        return 0

    closes, n_fresh = fetch_closes_window(sorted(real), as_of)
    record = {"date": str(as_of), "mode": mode,
              "generated_at": datetime.now(PT).isoformat(),
              "equity": equity, "stop_loss": cfg.stop_loss,
              "checks": [], "orders": [], "status": "ok"}
    if n_fresh == 0:
        record["status"] = "no_session"
        print(f"no bars dated {as_of} for any held name -- treating as "
              "a non-session (holiday); nothing checked", flush=True)
        _write_record(record)
        return 0

    checks = evaluate_stops(real, closes, cfg.stop_loss)
    record["checks"] = checks
    failed = [c for c in checks if c["status"] == "check_failed"]
    triggered = [c for c in checks if c["triggered"]]
    for c in checks:
        if c["status"] == "check_failed":
            print(f"  {c['symbol']}: CHECK FAILED ({c['reason']})",
                  flush=True)
        else:
            mark = "  <-- STOP TRIGGERED" if c["triggered"] else ""
            print(f"  {c['symbol']}: entry ${c['avg_entry_price']:,.2f} "
                  f"close ${c['close']:,.2f} ({c['ret_pct']:+.1%}) "
                  f"stop ${c['stop_level']:,.2f}{mark}", flush=True)

    if triggered and args.live:
        open_orders = broker.get_open_orders()
        if open_orders:
            record["status"] = "aborted_open_orders"
            print(f"ABORT: {len(open_orders)} open order(s) on the "
                  "account -- refusing to submit stop exits on top",
                  flush=True)
            _write_record(record)
            return 2
        for c in triggered:
            try:
                r = broker.submit_order(c["symbol"], "sell", qty=c["qty"])
                record["orders"].append(
                    {"symbol": c["symbol"], "qty": c["qty"],
                     "order_id": r.get("id"), "status": r.get("status")})
                print(f"  STOP EXIT submitted: {c['symbol']} "
                      f"qty {c['qty']} (order {r.get('id')})", flush=True)
            except Exception as e:  # noqa: BLE001 - record and continue
                record["orders"].append(
                    {"symbol": c["symbol"], "qty": c["qty"],
                     "error": str(e)})
                print(f"  STOP EXIT FAILED: {c['symbol']}: {e}", flush=True)
    elif triggered:
        for c in triggered:
            print(f"  WOULD EXIT (dry-run): {c['symbol']} qty {c['qty']}",
                  flush=True)

    submit_failed = [o for o in record["orders"] if "error" in o]
    if failed or submit_failed:
        record["status"] = "degraded"
    _write_record(record)
    print(f"stop check {as_of}: {len(triggered)} triggered, "
          f"{len(failed)} check-failed, status={record['status']}",
          flush=True)
    return 2 if record["status"] in ("degraded",) else 0


def _write_record(record: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"stoprun-{record['date']}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"record: {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
