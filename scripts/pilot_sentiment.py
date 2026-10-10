"""Sentiment orthogonality pilot (kill criterion for the sentiment desk).

For 50 large-cap tickers x 20 Friday review dates (2024-2026):
  - 28-day trailing GDELT news tone per (ticker, date)  [point-in-time: only
    articles published before the Friday close]
  - forward 1-week return from the local price panel
  - trailing 252-day momentum (proxy for the trend agent)

Reports:
  - corr(cross-sectional tone z-score, forward return) -- is there signal?
  - corr(tone z-score, momentum) -- or is tone just lagging price?
  - top/bottom tone-quintile forward-return spread.

Kill criterion: no forward-return signal beyond momentum -> stop the project.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.data.clients.gdelt import GDELTClient
from fund.data.db import get_engine
from fund.data.store import load_panel

MIN_ARTICLES = 5
N_DATES = 12  # reduced from 20: 600 queries is plenty for a pilot, halves API load
MAX_CONSEC_429 = 3  # abort the run if GDELT is throttling us; cron retries hourly


def review_fridays(n: int = N_DATES) -> list[date]:
    # Evenly spaced Fridays from 2024-01 to 2026-09.
    start, end = date(2024, 1, 5), date(2026, 9, 25)
    total = (end - start).days
    out = []
    for i in range(n):
        d = start + timedelta(days=int(i * total / (n - 1)))
        # roll back to Friday
        d -= timedelta(days=(d.weekday() - 4) % 7)
        out.append(d)
    return sorted(set(out))


def main() -> None:
    aliases: dict[str, str] = json.loads(
        (ROOT / "data" / "gdelt_aliases.json").read_text())
    tickers = sorted(aliases)
    print(f"pilot: {len(tickers)} tickers", flush=True)

    from fund.config import FundConfig
    engine = get_engine(str(ROOT / FundConfig().db_path))
    panel = load_panel(engine, tickers)
    closes = panel["close"]
    print(f"panel through {closes.index.max().date()}", flush=True)

    gdelt = GDELTClient(cache_dir=str(ROOT / "data" / "gdelt_cache"))
    dates = review_fridays()
    print(f"{len(dates)} review Fridays: {dates[0]} .. {dates[-1]}", flush=True)

    rows = []  # (date, ticker, tone_z, fwd_ret, momentum)
    consec_429 = 0
    for di, d in enumerate(dates):
        dstr = d.isoformat()
        tones: dict[str, float] = {}
        for t in tickers:
            q = aliases[t]
            start = (d - timedelta(days=28)).strftime("%Y%m%d")
            end = d.strftime("%Y%m%d")
            try:
                r = gdelt.tone(q, start, end)
                consec_429 = 0
            except Exception as e:  # noqa: BLE001 -- loud, keep going
                print(f"  GDELT miss {t} @ {dstr}: {e}", flush=True)
                if "429" in str(e):
                    consec_429 += 1
                    if consec_429 >= MAX_CONSEC_429:
                        print("  ABORT: GDELT throttling (3 consec 429s); "
                              "cron will retry later", flush=True)
                        raise SystemExit(42)
                else:
                    consec_429 = 0
                continue
            if r["avg_tone"] is not None and r["n_articles"] >= MIN_ARTICLES:
                tones[t] = r["avg_tone"]
        if len(tones) < 10:
            print(f"  {dstr}: only {len(tones)} tickers with coverage, skip",
                  flush=True)
            continue
        vals = np.array([tones[t] for t in tones])
        z = {t: (tones[t] - vals.mean()) / (vals.std() or 1.0) for t in tones}

        ts = pd.Timestamp(d)
        if ts not in closes.index:
            continue
        px = closes.loc[ts]
        # forward 1-week return: next Friday close if available else +5 sessions
        idx = closes.index.get_loc(ts)
        fwd_idx = min(idx + 5, len(closes.index) - 1)
        fwd_px = closes.iloc[fwd_idx]
        mom_px = closes.iloc[max(0, idx - 252)]
        for t in tones:
            if px.get(t) and fwd_px.get(t) and mom_px.get(t):
                fwd = fwd_px[t] / px[t] - 1.0
                mom = px[t] / mom_px[t] - 1.0
                rows.append((dstr, t, z[t], fwd, mom))
        print(f"  {dstr}: {len(tones)} tickers scored "
              f"(api: {gdelt.stats['requests']} req, "
              f"{gdelt.stats['cache_hits']} cache hits)", flush=True)

    df = pd.DataFrame(rows, columns=["date", "ticker", "tone_z", "fwd", "mom"])
    df.to_csv(ROOT / "results" / "sentiment_pilot.csv", index=False)
    print(f"\n{n} rows".replace("n", str(len(df))), flush=True)

    # Pooled correlations
    c_tf = df[["tone_z", "fwd"]].corr().iloc[0, 1]
    c_tm = df[["tone_z", "mom"]].corr().iloc[0, 1]
    c_mf = df[["mom", "fwd"]].corr().iloc[0, 1]
    print(f"corr(tone_z, fwd_1wk)   = {c_tf:+.3f}")
    print(f"corr(tone_z, momentum)  = {c_tm:+.3f}")
    print(f"corr(momentum, fwd_1wk) = {c_mf:+.3f}")

    # Quintile spread: top vs bottom tone_z forward returns
    df["q"] = df.groupby("date")["tone_z"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    spread = df.groupby("q")["fwd"].mean()
    print("\nmean fwd return by tone quintile (0=worst tone):")
    for q, v in spread.items():
        print(f"  Q{int(q)}: {v:+.4f}")
    print(f"\nQ4-Q0 spread: {spread.iloc[-1] - spread.iloc[0]:+.4f} per week")

    # Partial: does tone add anything beyond momentum? (cross-sectional residual)
    df["tone_resid"] = df["tone_z"] - df["mom"] * (
        df[["tone_z", "mom"]].corr().iloc[0, 1]
        * df["tone_z"].std() / (df["mom"].std() or 1.0))
    c_rf = df[["tone_resid", "fwd"]].corr().iloc[0, 1]
    print(f"corr(tone_resid_on_mom, fwd_1wk) = {c_rf:+.3f}")


if __name__ == "__main__":
    main()
