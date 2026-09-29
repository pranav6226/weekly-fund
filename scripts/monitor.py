#!/usr/bin/env python3
"""Phase 3 monitor: read-only daily health check of the Alpaca paper account.

Fetches the account (status, equity, buying power) and open positions,
compares them against the last paper_trade run's target weights
(results/paper_runs/latest.json), reports drift, and appends a daily
snapshot line to results/monitor.log (gitignored).

READ-ONLY BY CONSTRUCTION: the only broker calls in this file are
get_account() and get_positions() (both HTTP GET). There is no code path
that places, modifies, or cancels orders -- see tests/test_monitor.py,
which monkeypatches urlopen to explode on any POST/DELETE.

Usage:
    cd ~/workspace/weekly-fund && ./.venv/bin/python scripts/monitor.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.execution.broker import BrokerError, PaperBroker, alpaca_symbol  # noqa: E402

STATE_DIR = ROOT / "results" / "paper_runs"
LOG_PATH = ROOT / "results" / "monitor.log"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

DRIFT_THRESHOLD_PP = 0.01  # 1 percentage point
WOW_ALERT_THRESHOLD = -0.05  # equity down more than 5% week-over-week


# ------------------------------------------------------------------ targets
def load_targets(state_dir: Path = STATE_DIR) -> tuple[Optional[dict], Optional[str]]:
    """Return (targets {symbol: weight}, as_of) from latest.json, or (None, None)."""
    latest = state_dir / "latest.json"
    try:
        run = json.loads(latest.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    raw = run.get("targets") or {}
    targets = {alpaca_symbol(s): float(w) for s, w in raw.items()}
    return targets, run.get("as_of")


# --------------------------------------------------------------------- drift
def compute_drift(positions: dict[str, dict], targets: Optional[dict],
                  equity: float) -> dict[str, Any]:
    """Compare live positions against target weights.

    actual weight = market_value / equity. Flags:
      extra    - held but not in targets
      missing  - targeted but not held
      drifted  - held and targeted but |actual - target| > 1pp
    """
    pos = {str(s).upper(): v for s, v in positions.items()}
    tgt = targets or {}
    extra = sorted(s for s in pos if s not in tgt)
    missing = sorted(s for s in tgt if s not in pos)
    drifted = []
    if equity and equity > 0:
        for s in sorted(set(pos) & set(tgt)):
            actual = float(pos[s].get("market_value") or 0) / equity
            target = float(tgt[s])
            if abs(actual - target) > DRIFT_THRESHOLD_PP:
                drifted.append({"symbol": s, "actual": actual, "target": target})
    return {"extra": extra, "missing": missing, "drifted": drifted}


# ------------------------------------------------------------- week-over-week
def _parse_log_line(line: str) -> Optional[tuple[str, float]]:
    try:
        parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
        return parts["date"], float(parts["equity"])
    except (KeyError, ValueError):
        return None


def week_over_week(log_path: Path, equity: float, today: date) -> Optional[float]:
    """Equity % change vs the most recent snapshot at least 7 days old."""
    try:
        lines = log_path.read_text().splitlines()
    except OSError:
        return None
    week_ago = date.fromordinal(today.toordinal() - 7).isoformat()
    best: Optional[tuple[str, float]] = None
    for line in lines:
        parsed = _parse_log_line(line)
        if not parsed:
            continue
        d, e = parsed
        if d <= week_ago and (best is None or d > best[0]):
            best = (d, e)
    if best is None or best[1] <= 0:
        return None
    return equity / best[1] - 1.0


def day_over_day(log_path: Path, equity: float, today: date) -> Optional[tuple[float, float]]:
    """Equity $ and % change vs the most recent snapshot from a previous day.

    Skips any lines dated today (compare on the calendar-date portion only,
    so intraday entries like the fill-check's ISO timestamp don't count).
    Returns (dollar_change, pct_change) or None when there is no prior day.
    """
    try:
        lines = log_path.read_text().splitlines()
    except OSError:
        return None
    today_s = today.isoformat()
    best: Optional[tuple[str, float]] = None
    for line in lines:
        parsed = _parse_log_line(line)
        if not parsed:
            continue
        d, e = parsed
        day = d[:10]  # plain dates already are YYYY-MM-DD; ISO timestamps truncate
        if day >= today_s:
            continue
        if best is None or day > best[0][:10]:
            best = (d, e)
    if best is None or best[1] <= 0:
        return None
    prev = best[1]
    return equity - prev, equity / prev - 1.0


# ------------------------------------------------------------------- snapshot
def append_snapshot(log_path: Path, result: dict) -> None:
    def syms(xs):
        return ";".join(xs) if xs else "-"

    drift = ";".join(
        f"{d['symbol']}:{d['actual']:.4f}:{d['target']:.4f}"
        for d in result["drifted"]
    ) or "-"
    wow = f"{result['wow']:.4f}" if result["wow"] is not None else "-"
    dod = (
        f"{result['dod'][0]:.2f}:{result['dod'][1]:.4f}"
        if result["dod"] is not None else "-"
    )
    line = (
        f"date={result['date']} equity={result['equity']:.2f} "
        f"npos={result['n_positions']} ntargets={result['n_targets']} "
        f"extra={syms(result['extra'])} missing={syms(result['missing'])} "
        f"drift={drift} status={result['status']} wow={wow} dod={dod}\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(line)


# ---------------------------------------------------------------------- check
def run_check(broker, *, state_dir: Path = STATE_DIR,
              log_path: Path = LOG_PATH, today: Optional[date] = None) -> dict:
    """Full read-only check. Only GET calls happen inside. Returns a result dict."""
    today = today or datetime.now(LOCAL_TZ).date()
    # --- the only two broker calls in the entire monitor (both HTTP GET) ---
    account = broker.get_account()
    positions = broker.get_positions()
    # -------------------------------------------------------------------------
    equity = float(account.get("equity") or 0)
    status = str(account.get("status") or "UNKNOWN")

    targets, targets_as_of = load_targets(state_dir)
    drift = compute_drift(positions, targets, equity)
    wow = week_over_week(log_path, equity, today)
    dod = day_over_day(log_path, equity, today)

    anomalies: list[str] = []
    if status != "ACTIVE":
        anomalies.append(f"ACCOUNT_STATUS={status}")
    if targets is None:
        anomalies.append("NO_TARGETS_FILE")
    if drift["extra"]:
        anomalies.append(f"EXTRA_POSITIONS={len(drift['extra'])}")
    if drift["missing"]:
        anomalies.append(f"MISSING_POSITIONS={len(drift['missing'])}")
    if drift["drifted"]:
        anomalies.append(f"WEIGHT_DRIFT={len(drift['drifted'])}")
    if wow is not None and wow < WOW_ALERT_THRESHOLD:
        anomalies.append(f"EQUITY_DOWN_WOW={wow:.1%}")

    result = {
        "date": today.isoformat(),
        "status": status,
        "equity": equity,
        "buying_power": float(account.get("buying_power") or 0),
        "cash": float(account.get("cash") or 0),
        "n_positions": len(positions),
        "positions": positions,
        "targets_as_of": targets_as_of,
        "n_targets": len(targets) if targets else 0,
        "extra": drift["extra"],
        "missing": drift["missing"],
        "drifted": drift["drifted"],
        "wow": wow,
        "dod": dod,
        "anomalies": anomalies,
        "healthy": not anomalies,
    }
    append_snapshot(log_path, result)
    return result


def _fmt_list(xs: list[str], limit: int = 6) -> str:
    if not xs:
        return "-"
    shown = ", ".join(xs[:limit])
    return shown + (f" (+{len(xs) - limit} more)" if len(xs) > limit else "")


def format_summary(r: dict) -> str:
    lines = [
        f"Fund monitor {r['date']} -- Alpaca paper account",
        f"  account {r['status']} | equity ${r['equity']:,.2f} | "
        f"buying power ${r['buying_power']:,.2f}",
        f"  positions: {r['n_positions']} open vs {r['n_targets']} targets"
        + (f" (run {r['targets_as_of']})" if r['targets_as_of'] else " (no targets file)"),
    ]
    if r["extra"] or r["missing"] or r["drifted"]:
        drifted_str = (
            ", ".join(
                f"{d['symbol']} {d['actual']:.1%} vs {d['target']:.1%}"
                for d in r["drifted"]
            )
            or "-"
        )
        lines.append(
            f"  drift: extra [{_fmt_list(r['extra'])}]; "
            f"missing [{_fmt_list(r['missing'])}]; "
            f"drifted>1pp [{drifted_str}]"
        )
    else:
        lines.append("  drift: none")
    lines.append(
        f"  week-over-week equity: {r['wow']:+.1%}"
        if r["wow"] is not None else "  week-over-week equity: n/a (no history yet)"
    )
    lines.append(
        f"  day-over-day equity: {r['dod'][0]:+,.2f} ({r['dod'][1]:+.2%})"
        if r["dod"] is not None else "  day-over-day equity: n/a (no prior-day snapshot)"
    )
    if r["anomalies"]:
        lines.append("  anomalies: " + "; ".join(r["anomalies"]))
    return "\n".join(lines)


def main() -> int:
    try:
        broker = PaperBroker()  # keys from env or repo-root .env; never logged
        result = run_check(broker)
    except BrokerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(format_summary(result), flush=True)
    print("STATUS: " + ("HEALTHY" if result["healthy"] else "ANOMALY"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
