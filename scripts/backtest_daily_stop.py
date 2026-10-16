"""A/B backtest: daily stop enforcement vs weekly-only stops.

The v3 simulator has ALWAYS scanned stops on every daily close
(simulator step 3) -- so the validated v3 numbers (+273.7% / 21.7%
CAGR / -27.8% max DD) assume daily stop enforcement. The live loop
today checks stops only at the Saturday review, through
construct_targets. This run quantifies what that difference is worth:

  variant A  daily_stop=True   the v3 baseline as validated
  variant B  daily_stop=False  the live loop's current behavior
             (review-time stop exits still apply via construct_targets)

Everything else is identical v3 wiring (same score/regime/fuse
functions as scripts/run_backtest_v3.py). Desk scores are
path-independent, so score_fn is memoized per review date and both
simulations share one scoring pass.

Usage: python scripts/backtest_daily_stop.py [--start ...] [--end ...]
Outputs: results/backtest_daily_stop/{metrics.json, equity_daily.csv,
         equity_weekly.csv, trades_daily.csv, trades_weekly.csv}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.backtest.simulator import run_backtest  # noqa: E402
from fund.config import FundConfig  # noqa: E402
from fund.data.clients.free_stack import FreeDataStack  # noqa: E402
from fund.data.db import get_engine  # noqa: E402
from fund.data.store import load_panel  # noqa: E402
from fund.decision.fusion import (  # noqa: E402
    RegimeResult,
    blend_regime,
    fuse_desks,
    market_regime,
)
from fund.mandates.schema import load_mandate  # noqa: E402
from fund.research.desks import score_desks  # noqa: E402


def review_execution_dates(closes: pd.DataFrame, start: str,
                           end: str) -> set:
    """Dates on which weekly-review orders execute (the trading day
    after each ISO week's last trading day) -- used to tell review
    exits apart from stop exits in the trade log."""
    mask = (closes.index >= pd.to_datetime(start)) & \
           (closes.index <= pd.to_datetime(end))
    dates = list(closes.index[mask])
    weeks: dict[tuple, list] = {}
    for d in dates:
        iso = d.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(d)
    pos_of = {d: i for i, d in enumerate(dates)}
    out = set()
    for w in weeks.values():
        nxt = pos_of[w[-1]] + 1
        if nxt < len(dates):
            out.add(dates[nxt].date())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-09-25")
    args = ap.parse_args()

    cfg = FundConfig(backtest_start=args.start, backtest_end=args.end)
    mandate = load_mandate(str(ROOT / "mandates" / "core-4desk.yaml"))
    desk_weights = {d.name: d.weight for d in mandate.desks}
    print(f"mandate: core-4desk, weights={desk_weights}", flush=True)

    engine = get_engine(str(ROOT / cfg.db_path))
    print("loading panel...", flush=True)
    panel = load_panel(engine)
    closes_full = panel["close"]
    print(f"panel: {closes_full.shape[0]} days x "
          f"{closes_full.shape[1]} tickers", flush=True)

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False),
                       uni["GICS Sector"]))

    stack = FreeDataStack(cache_root=str(ROOT / "data"))
    state = {"macro_f": 1.0}
    score_cache: dict = {}

    def score_fn(closes_f, volumes_f, as_of, cfg_, sectors_):
        key = pd.Timestamp(as_of)
        if key in score_cache:
            scores, macro_f = score_cache[key]
            state["macro_f"] = macro_f  # replay the side effect
            return scores
        closes = closes_f.loc[:as_of]
        volumes = volumes_f.loc[:as_of]
        stack.set_closes(closes)
        scores, macro_f = score_desks(stack, closes, volumes, as_of,
                                      mandate, sectors_)
        score_cache[key] = (scores, macro_f)
        state["macro_f"] = macro_f
        return scores

    def regime_fn(closes_f, as_of, cfg_):
        price = market_regime(closes_f, as_of, cfg_)
        f = blend_regime(price.factor, state["macro_f"], blend=0.5,
                         floor=cfg_.regime_floor)
        return RegimeResult(round(f, 3), price.label,
                            f"{price.rationale}|macro={state['macro_f']:.2f}")

    def fuse_fn(scored, cfg_, factor):
        return fuse_desks(scored, desk_weights, factor)

    exec_dates = review_execution_dates(closes_full, args.start, args.end)
    out_dir = ROOT / "results" / "backtest_daily_stop"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"window": [args.start, args.end], "variants": {}}

    for label, daily in (("daily", True), ("weekly", False)):
        t0 = time.time()
        print(f"\n=== variant {label} (daily_stop={daily}) ===", flush=True)
        res = run_backtest(panel, sectors, score_fn, cfg, args.start,
                           args.end, regime_fn=regime_fn, fuse_fn=fuse_fn,
                           daily_stop=daily)
        sells = res.trades[res.trades["side"] == "SELL"] \
            if not res.trades.empty else res.trades
        stop_exits = int(sum(1 for d in sells["date"]
                             if d not in exec_dates)) \
            if not sells.empty else 0
        m = dict(res.metrics)
        m["stop_exits"] = stop_exits
        m["reviews"] = res.review_count
        summary["variants"][label] = m
        res.equity.to_csv(out_dir / f"equity_{label}.csv", header=["equity"])
        res.trades.to_csv(out_dir / f"trades_{label}.csv", index=False)
        print(f"variant {label} done in {time.time() - t0:.0f}s: {m}",
              flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    a, b = summary["variants"]["daily"], summary["variants"]["weekly"]
    print("\n==== daily stops vs weekly-only stops ====")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "trades",
              "round_trips", "win_rate", "turnover_x", "stop_exits",
              "final_equity"):
        print(f"{k:<14} daily={a.get(k)!s:>12}  weekly={b.get(k)!s:>12}")
    print(f"\nwrote {out_dir}/metrics.json + equity/trades CSVs", flush=True)


if __name__ == "__main__":
    main()
