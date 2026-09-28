"""Compare v1 (momentum+lowvol heuristic) vs v2 (swarm + Jev-shaped fusion).

Usage:
    python scripts/compare_v1_v2.py [--start 2020-01-01] [--end 2026-09-25]

Runs both scorers through the IDENTICAL simulator and portfolio
constructor, so any difference is the strategy, not the plumbing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.backtest.simulator import compute_metrics, run_backtest  # noqa: E402
from fund.config import FundConfig  # noqa: E402
from fund.data.db import get_engine  # noqa: E402
from fund.data.store import load_panel  # noqa: E402
from fund.research.agents import score_universe, score_universe_v1  # noqa: E402


def load_spy(start: str, end: str) -> pd.Series:
    import yfinance as yf

    df = yf.download("SPY", start="2019-12-01", end=end,
                     auto_adjust=True, progress=False)
    s = df["Close"]["SPY"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end].dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-09-25")
    args = ap.parse_args()

    cfg = FundConfig(backtest_start=args.start, backtest_end=args.end)
    engine = get_engine(str(ROOT / cfg.db_path))
    print("loading panel...", flush=True)
    panel = load_panel(engine)
    print(f"panel: {panel['close'].shape[0]} days x {panel['close'].shape[1]} tickers")

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False), uni["GICS Sector"]))

    results = {}
    for name, fn in [("v1", score_universe_v1), ("v2.1", score_universe)]:
        t0 = time.time()
        res = run_backtest(panel, sectors, fn, cfg, args.start, args.end)
        print(f"{name}: {res.review_count} reviews, {len(res.trades)} trades, "
              f"{time.time()-t0:.1f}s")
        results[name] = res

    spy = load_spy(args.start, args.end)
    spy_eq = spy / spy.iloc[0] * cfg.initial_capital
    spy_m = compute_metrics(spy_eq, pd.DataFrame())

    print(f"\n==== v1 vs v2.1 vs SPY ({args.start} -> {args.end}) ====")
    keys = list(results["v2.1"].metrics)
    rows = [("metric", "v1", "v2.1", "spy")]
    for k in keys:
        rows.append((k, results["v1"].metrics[k], results["v2.1"].metrics[k],
                     spy_m.get(k)))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]!s:>12}  {r[2]!s:>12}  {r[3]!s:>12}")

    out = ROOT / cfg.results_dir
    out.mkdir(exist_ok=True)
    results["v1"].equity.to_csv(out / "equity_v1.csv", header=["equity"])
    results["v2.1"].equity.to_csv(out / "equity_v2_1.csv", header=["equity"])
    spy_eq.to_csv(out / "equity_spy.csv", header=["equity"])
    results["v1"].trades.to_csv(out / "trades_v1.csv", index=False)
    results["v2.1"].trades.to_csv(out / "trades_v2_1.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results["v1"].equity.index, results["v1"].equity.values,
                label="v1 momentum+lowvol", alpha=0.8)
        ax.plot(results["v2.1"].equity.index, results["v2.1"].equity.values,
                label="v2.1 regime swarm", linewidth=1.5)
        ax.plot(spy_eq.index, spy_eq.values, label="SPY", alpha=0.6)
        ax.legend()
        ax.set_title("Walk-forward backtest: v1 vs v2 vs SPY (net of costs)")
        fig.tight_layout()
        fig.savefig(out / "equity_v1_v2.png")
        print(f"\nwrote {out}/equity_v1_v2.png + csvs")
    except ImportError:
        print("(matplotlib not installed; skipping chart)")


if __name__ == "__main__":
    main()
