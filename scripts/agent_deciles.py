"""Decile-spread diagnostic: for each agent, sort liquid tickers into
deciles by z-score at each review; report mean forward-63d return of the
top decile minus the bottom decile. This matches how the portfolio
actually uses scores (it only holds the top ~20 names)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig
from fund.data.db import get_engine
from fund.data.store import load_panel
from fund.research.agents import AGENTS, liquid_tickers, score_universe_v1


def decile_spread(z: pd.Series, fwd: pd.Series) -> float:
    idx = z.dropna().index.intersection(fwd.dropna().index)
    if len(idx) < 100:
        return np.nan
    z, f = z[idx], fwd[idx]
    try:
        dec = pd.qcut(z, 10, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    top = f[dec == dec.max()].mean()
    bot = f[dec == dec.min()].mean()
    return float(top - bot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=63)
    args = ap.parse_args()
    cfg = FundConfig()
    engine = get_engine(str(ROOT / cfg.db_path))
    panel = load_panel(engine)
    closes, volumes = panel["close"], panel["volume"]

    all_dates = closes.index
    review_days = [d for d in all_dates if d.weekday() == 4]
    review_days = [d for d in review_days
                   if d >= pd.Timestamp(cfg.backtest_start)
                   and d <= pd.Timestamp(cfg.backtest_end)]
    review_days = [d for d in review_days
                   if all_dates.get_loc(d) + args.fwd + 1 < len(all_dates)]

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False),
                       uni["GICS Sector"]))

    spreads: dict[str, list[float]] = {n: [] for n in list(AGENTS) + ["v1-blend"]}
    for di, d in enumerate(review_days):
        i = all_dates.get_loc(d)
        fwd = closes.iloc[i + args.fwd] / closes.iloc[i + 1] - 1
        liquid = liquid_tickers(closes, volumes, d, cfg)
        fwd = fwd[[t for t in liquid if t in fwd.index]].dropna()
        for name, fn in AGENTS.items():
            if name == "trend":
                z = pd.Series(fn(closes, d, cfg, sectors))
            elif name == "flow":
                z = pd.Series(fn(closes, volumes, d, cfg))
            else:
                z = pd.Series(fn(closes, d, cfg))
            s = decile_spread(z, fwd)
            if not np.isnan(s):
                spreads[name].append(s)
        v1 = {s.ticker: s.score for s in score_universe_v1(closes, volumes, d, cfg)}
        s = decile_spread(pd.Series(v1), fwd)
        if not np.isnan(s):
            spreads["v1-blend"].append(s)
        if di % 50 == 0:
            print(f"  {d.date()} ({di+1}/{len(review_days)})", flush=True)

    print(f"\n==== top-minus-bottom decile, fwd {args.fwd}d (per review, then annualized x4) ====")
    print(f"{'agent':<12} {'n':>5} {'mean_sprd':>10} {'ann':>8} {'t-stat':>8}")
    for name, vals in spreads.items():
        v = np.array(vals)
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
        print(f"{name:<12} {len(v):>5} {v.mean():>10.4f} {v.mean()*4:>8.2f} {t:>8.2f}")


if __name__ == "__main__":
    main()
