"""Batch download of daily OHLCV history via yfinance into the fund's SQLite store.

Universe comes from data/universe.csv (Symbol, GICS Sector). Prices are
total-return adjusted (auto_adjust=True) so dividends/splits are reflected
in the price series -- the backtester therefore needs no separate dividend
bookkeeping.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Sandbox egress goes through the proxy.
for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fund.data.db import get_engine, init_db  # noqa: E402
from fund.data.store import save_bars  # noqa: E402

START = "2018-01-01"
CHUNK = 100


def yf_symbol(sym: str) -> str:
    """Wikipedia-style 'BRK.B' -> yfinance-style 'BRK-B'."""
    return sym.replace(".", "-")


def download_chunk(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return df


def normalize(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Wide multi-ticker frame -> long (ticker, date, o/h/l/c/volume)."""
    if df.empty:
        return pd.DataFrame()
    # yfinance >= 0.2.32: columns are MultiIndex (field, ticker) by default
    # in newer versions; handle both orientations.
    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Open" in lvl0:  # (field, ticker)
            fields = df.columns.get_level_values(0)
            tkrs = df.columns.get_level_values(1)
            for t in tickers:
                sub = df.xs(t, level=1, axis=1)
                sub = sub.rename(columns=str.lower)
                sub["ticker"] = t
                frames.append(sub)
        else:  # (ticker, field)
            for t in tickers:
                try:
                    sub = df.xs(t, level=0, axis=1)
                except KeyError:
                    continue
                sub = sub.rename(columns=str.lower)
                sub["ticker"] = t
                frames.append(sub)
    else:  # single ticker
        sub = df.rename(columns=str.lower)
        sub["ticker"] = tickers[0]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames)
    long.index.name = "date"
    long = long.reset_index()
    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    long = long[[c for c in keep if c in long.columns]]
    return long.dropna(subset=["close"])


def main() -> None:
    end = time.strftime("%Y-%m-%d")
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    tickers = [yf_symbol(s) for s in uni["Symbol"].tolist()]
    print(f"universe: {len(tickers)} tickers, {START} -> {end}")

    db_path = ROOT / "data" / "market.db"
    engine = get_engine(str(db_path))
    init_db(engine)

    all_long = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        print(f"[{i}/{len(tickers)}] downloading {len(chunk)} tickers...", flush=True)
        try:
            df = download_chunk(chunk, START, end)
        except Exception as e:  # noqa: BLE001
            print(f"  chunk failed: {e}")
            continue
        long = normalize(df, chunk)
        print(f"  rows: {len(long)}")
        all_long.append(long)
        time.sleep(2)

    if all_long:
        full = pd.concat(all_long, ignore_index=True)
        n = save_bars(engine, full)
        print(f"saved {n} bars for {full['ticker'].nunique()} tickers -> {db_path}")
    else:
        print("nothing downloaded")


if __name__ == "__main__":
    main()
