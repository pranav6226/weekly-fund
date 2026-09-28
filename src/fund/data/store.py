"""Read/write helpers for the market-data store."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text


def save_bars(engine, df_long: pd.DataFrame) -> int:
    """Append long-format bars (ticker,date,open,high,low,close,volume).

    Callers rebuild the DB from scratch; no upsert logic needed for v1.
    """
    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM daily_bars"))
    df.to_sql("daily_bars", engine, if_exists="append", index=False)
    return len(df)


def load_panel(engine, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return wide OHLCV panels: {field: DataFrame(date index, ticker columns)}."""
    q = "SELECT ticker, date, open, high, low, close, volume FROM daily_bars"
    df = pd.read_sql(q, engine, parse_dates=["date"])
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "date"])
    panel = {}
    for field in ("open", "high", "low", "close", "volume"):
        wide = df.pivot(index="date", columns="ticker", values=field)
        wide.index = pd.to_datetime(wide.index)
        panel[field] = wide
    return panel


def coverage(engine) -> pd.DataFrame:
    q = ("SELECT ticker, COUNT(*) n, MIN(date) d0, MAX(date) d1 "
         "FROM daily_bars GROUP BY ticker ORDER BY ticker")
    return pd.read_sql(q, engine)
