"""Regime-conditional decile spreads: does the defensive rotation prior
hold? i.e. is qualvol's top-minus-bottom spread positive in risk-off
regimes and negative in risk-on? Diagnostic only -- tests the prior,
does not fit anything."""
from __future__ import annotations

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
from fund.decision.fusion import market_regime
from fund.research.agents import AGENTS, liquid_tickers


def decile_spread(z: pd.Series, fwd: pd.Series) -> float:
    idx = z.dropna().index.intersection(fwd.dropna().index)
    if len(idx) < 100:
        return np.nan
    z, f = z[idx], fwd[idx]
    try:
        dec = pd.qcut(z, 10, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    return float(f[dec == dec.max()].mean() - f[dec == dec.min()].mean())


def main() -> None:
    cfg = FundConfig()
    engine = get_engine(str(ROOT / cfg.db_path))
    panel = load_panel(engine)
    closes, volumes = panel["close"], panel["volume"]
    FWD = 63

    all_dates = closes.index
    review_days = [d for d in all_dates if d.weekday() == 4]
    review_days = [d for d in review_days
                   if d >= pd.Timestamp(cfg.backtest_start)
                   and d <= pd.Timestamp(cfg.backtest_end)]
    review_days = [d for d in review_days
                   if all_dates.get_loc(d) + FWD + 1 < len(all_dates)]

    uni = pd.read_csv(ROOT / cfg.universe_csv)
    sectors = dict(zip(uni["Symbol"].str.replace(".", "-", regex=False),
                       uni["GICS Sector"]))

    buckets: dict[str, dict[str, list[float]]] = {
        "risk-on": {n: [] for n in AGENTS},
        "risk-off": {n: [] for n in AGENTS},
    }
    for di, d in enumerate(review_days):
        reg = market_regime(closes, d, cfg)
        bucket = "risk-off" if reg.factor <= 0.6 else ("risk-on" if reg.factor >= 0.99 else None)
        if bucket is None:
            continue
        i = all_dates.get_loc(d)
        fwd = closes.iloc[i + FWD] / closes.iloc[i + 1] - 1
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
                buckets[bucket][name].append(s)

    for bucket in ("risk-on", "risk-off"):
        print(f"\n==== {bucket} (fwd {FWD}d top-bottom decile) ====")
        for name, vals in buckets[bucket].items():
            v = np.array(vals)
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
            print(f"  {name:<10} n={len(v):>3}  mean={v.mean():+.4f}  t={t:+.2f}")


if __name__ == "__main__":
    main()
