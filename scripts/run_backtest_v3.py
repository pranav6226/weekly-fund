"""Run the v3 walk-forward backtest: 4-desk swarm on the free data stack.

Usage:
    python scripts/run_backtest_v3.py [--start 2020-01-01] [--end 2026-09-25] [--max-tickers 0]

Scoring per Friday review:
  technicals   deterministic agents (v2.1: trend/reversal/flow, regime-interpolated)
  fundamentals SEC XBRL 10-K metrics, filing-date-gated
  filings      SEC 8-Ks: PEAD (2-day CAR surprise proxy) + abnormal 8-K attention
  macro        FRED yields/CPI/unemployment -> regime factor blended 50/50
               with the price regime (degrades to price-only without FRED_API_KEY)

Fusion: fuse_desks (two-level: regime-interpolated technicals mix inside,
fixed mandate priors across desks, Phi(z) -> convictions).

Point-in-time discipline: closes/volumes are sliced to as_of before any
desk sees them; the SEC stack hands desks dated rows and the desks filter
on filing_date <= as_of FIRST.
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
    ap.add_argument("--max-tickers", type=int, default=0,
                    help="limit universe size (0 = full)")
    args = ap.parse_args()

    cfg = FundConfig(backtest_start=args.start, backtest_end=args.end)
    mandate = load_mandate(str(ROOT / "mandates" / "core-4desk.yaml"))
    desk_weights = {d.name: d.weight for d in mandate.desks}
    print(f"mandate: core-4desk, weights={desk_weights}", flush=True)

    engine = get_engine(str(ROOT / cfg.db_path))
    print("loading panel...", flush=True)
    panel = load_panel(engine)
    closes_full, volumes_full = panel["close"], panel["volume"]
    if args.max_tickers:
        keep = closes_full.columns[: args.max_tickers]
        closes_full, volumes_full = closes_full[keep], volumes_full[keep]
    print(f"panel: {closes_full.shape[0]} days x {closes_full.shape[1]} tickers",
          flush=True)

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False),
                       uni["GICS Sector"]))

    stack = FreeDataStack(cache_root=str(ROOT / "data"))
    state = {"macro_f": 1.0, "macro_note": ""}

    def score_fn_v3(closes_f, volumes_f, as_of, cfg_, sectors_):
        # strictly point-in-time: desks never see a print after as_of
        closes = closes_f.loc[:as_of]
        volumes = volumes_f.loc[:as_of]
        stack.set_closes(closes)
        scores, macro_f = score_desks(stack, closes, volumes, as_of,
                                      mandate, sectors_)
        state["macro_f"] = macro_f
        return scores

    def regime_fn_v3(closes_f, as_of, cfg_):
        price = market_regime(closes_f, as_of, cfg_)
        f = blend_regime(price.factor, state["macro_f"], blend=0.5,
                         floor=cfg_.regime_floor)
        return RegimeResult(round(f, 3), price.label,
                            f"{price.rationale}|macro={state['macro_f']:.2f}")

    def fuse_fn_v3(scored, cfg_, factor):
        return fuse_desks(scored, desk_weights, factor)

    t0 = time.time()
    res = run_backtest(panel, sectors, score_fn_v3, cfg, args.start, args.end,
                       regime_fn=regime_fn_v3, fuse_fn=fuse_fn_v3)
    print(f"simulated {res.review_count} weekly reviews in {time.time()-t0:.1f}s")

    spy = load_spy(args.start, args.end)
    spy_eq = spy / spy.iloc[0] * cfg.initial_capital
    spy_metrics = compute_metrics(spy_eq, pd.DataFrame())

    print("\n==== v3 4-desk swarm (free stack) vs SPY buy-and-hold ====")
    print(f"window: {args.start} -> {args.end} | universe: {closes_full.shape[1]} tickers")
    rows = [("metric", "fund-v3", "spy")]
    for k in res.metrics:
        rows.append((k, res.metrics[k], spy_metrics.get(k)))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]!s:>12}  {r[2]!s:>12}")

    out = ROOT / cfg.results_dir
    out.mkdir(exist_ok=True)
    res.equity.to_csv(out / "equity_v3.csv", header=["equity"])
    spy_eq.to_csv(out / "equity_spy_v3.csv", header=["equity"])
    res.trades.to_csv(out / "trades_v3.csv", index=False)
    print(f"\nwrote {out}/equity_v3.csv, equity_spy_v3.csv, trades_v3.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(res.equity.index, res.equity.values, label="fund v3")
        ax.plot(spy_eq.index, spy_eq.values, label="SPY", alpha=0.7)
        ax.legend()
        ax.set_title("v3 walk-forward backtest (net of 5bps/side slippage)")
        fig.tight_layout()
        fig.savefig(out / "equity_v3.png")
        print(f"wrote {out}/equity_v3.png")
    except ImportError:
        print("(matplotlib not installed; skipping chart)")


if __name__ == "__main__":
    main()
