"""Per-agent Information Coefficient diagnostic.

For each weekly review date: rank-correlation (Spearman) between each
agent's z-score at T and the ticker's forward 5-day return (T+1 -> T+6,
i.e. the return the NEXT review's portfolio would capture). Fully
walk-forward. Reports mean IC and t-stat per agent.

A positive, significant IC means the agent ranks winners above losers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import numpy as np
import pandas as pd


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy."""
    r = pd.DataFrame({"a": a, "b": b}).rank()
    return float(r["a"].corr(r["b"]))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.config import FundConfig
from fund.data.db import get_engine
from fund.data.store import load_panel
from fund.research.agents import AGENTS, liquid_tickers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=5, help="forward return window (trading days)")
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
    # need forward window available
    review_days = [d for d in review_days
                   if all_dates.get_loc(d) + args.fwd + 1 < len(all_dates)]

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False),
                       uni["GICS Sector"]))

    ics: dict[str, list[float]] = {name: [] for name in AGENTS}
    ics["v1-blend"] = []

    for di, d in enumerate(review_days):
        i = all_dates.get_loc(d)
        fwd = closes.iloc[i + args.fwd] / closes.iloc[i + 1] - 1  # fwd-window return
        liquid = liquid_tickers(closes, volumes, d, cfg)
        fwd = fwd.dropna()
        common = [t for t in liquid if t in fwd.index]
        if len(common) < 50:
            continue
        f = fwd[common]
        for name, fn in AGENTS.items():
            try:
                if name == "trend":
                    z = pd.Series(fn(closes, d, cfg, sectors))
                elif name == "flow":
                    z = pd.Series(fn(closes, volumes, d, cfg))
                else:
                    z = pd.Series(fn(closes, d, cfg))
            except Exception:
                continue
            z = z.dropna()
            idx = [t for t in common if t in z.index]
            if len(idx) < 50:
                continue
            rho = spearman(z[idx], f[idx])
            if not np.isnan(rho):
                ics[name].append(rho)
        if di % 50 == 0:
            print(f"  {d.date()} ({di+1}/{len(review_days)})", flush=True)

    print(f"\n==== mean IC per agent (fwd {args.fwd}d return) ====")
    print(f"{'agent':<12} {'n':>5} {'mean_IC':>9} {'t-stat':>8}")
    for name, vals in ics.items():
        v = np.array(vals)
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
        print(f"{name:<12} {len(v):>5} {v.mean():>9.4f} {t:>8.2f}")


if __name__ == "__main__":
    main()
