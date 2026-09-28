"""Run the v1 walk-forward backtest and print the report.

Usage:
    python scripts/run_backtest.py [--start 2020-01-01] [--end 2026-09-25]
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
from fund.research.agents import score_universe  # noqa: E402


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

    t0 = time.time()
    res = run_backtest(panel, sectors, score_universe, cfg, args.start, args.end)
    print(f"simulated {res.review_count} weekly reviews in {time.time()-t0:.1f}s")

    spy = load_spy(args.start, args.end)
    spy_eq = spy / spy.iloc[0] * cfg.initial_capital
    spy_metrics = compute_metrics(spy_eq, pd.DataFrame())

    print("\n==== v1 momentum+lowvol vs SPY buy-and-hold ====")
    print(f"window: {args.start} -> {args.end} | universe: S&P 500 (current constituents)")
    rows = [("metric", "fund-v1", "spy")]
    for k in res.metrics:
        rows.append((k, res.metrics[k], spy_metrics.get(k)))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]!s:>12}  {r[2]!s:>12}")

    out = ROOT / cfg.results_dir
    out.mkdir(exist_ok=True)
    res.equity.to_csv(out / "equity_fund.csv", header=["equity"])
    spy_eq.to_csv(out / "equity_spy.csv", header=["equity"])
    res.trades.to_csv(out / "trades.csv", index=False)
    print(f"\nwrote {out}/equity_fund.csv, equity_spy.csv, trades.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(res.equity.index, res.equity.values, label="fund v1")
        ax.plot(spy_eq.index, spy_eq.values, label="SPY", alpha=0.7)
        ax.legend()
        ax.set_title("Walk-forward backtest (net of 5bps/side slippage)")
        fig.tight_layout()
        fig.savefig(out / "equity.png")
        print(f"wrote {out}/equity.png")
    except ImportError:
        print("(matplotlib not installed; skipping chart)")


if __name__ == "__main__":
    main()
