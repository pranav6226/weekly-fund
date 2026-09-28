"""FreeDataStack -- the fund's data layer on 100% free sources.

Replaces the paid Financial Datasets client (deleted). Every source here
is free; only FRED asks for a (free, no-payment) API key:

  prices       yfinance (already in the stack, unchanged)
  fundamentals SEC EDGAR XBRL companyfacts -- as-reported 10-K facts with
               `filed` dates; ratios computed here, gated on filing_date
  filings      EDGAR 8-Ks -- Item 2.02 (earnings) identified via the
               full-text search API; all other 8-Ks as attention events
  macro        FRED -- DGS10/DGS2, CPIAUCSL, UNRATE (free key in .env)

Interface is deliberately narrow: metrics(), filing_events(), and the
three macro histories. Desks filter on dates FIRST -- the stack hands
them dated rows, the walk-forward discipline lives in the desks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class FreeDataStack:
    def __init__(self, cache_root: str | Path = "data",
                 closes: pd.DataFrame | None = None):
        from fund.data.clients.sec import SECClient
        self._sec = SECClient(cache_dir=Path(cache_root) / "sec_cache")
        self._fred = None
        self._fred_cache_dir = Path(cache_root) / "fred_cache"
        self._closes = closes

    def set_closes(self, closes: pd.DataFrame) -> None:
        """Prices for as-of valuation ratios (P/E, FCF yield)."""
        self._closes = closes

    # ------------------------------------------------------------ fundamentals
    def _price_on_or_before(self, ticker: str, d) -> float | None:
        if self._closes is None or ticker not in self._closes.columns:
            return None
        try:
            px = self._closes[ticker].loc[:pd.to_datetime(d)].dropna()
            return float(px.iloc[-1]) if len(px) else None
        except (KeyError, IndexError, TypeError):
            return None

    def metrics(self, ticker: str, limit: int = 8) -> list[dict]:
        """Annual metric rows, newest first, each with a filing_date.

        Ratios: P/E and FCF yield need a price, so they use the close on
        the 10-K filing date (never after -- no lookahead). Rows without a
        price still carry the accounting ratios; the desk scores
        completeness instead of imputing.
        """
        facts = self._sec.annual_facts(ticker, limit=limit + 1)
        rows = []
        for i, f in enumerate(facts[:limit]):
            prev = facts[i + 1] if i + 1 < len(facts) else None
            px = self._price_on_or_before(ticker, f["filing_date"]) \
                if f["filing_date"] else None
            eps, peps = f.get("eps_diluted"), (prev or {}).get("eps_diluted")
            rev, prev_rev = f.get("revenue"), (prev or {}).get("revenue")
            fcf = ((f.get("ocf") or 0) - (f.get("capex") or 0)
                   if f.get("ocf") is not None else None)
            shares = f.get("diluted_shares")
            rows.append({
                "fy": f["fy"],
                "filing_date": f["filing_date"],
                "price_to_earnings_ratio":
                    (px / eps) if px and eps and eps > 0 else None,
                "free_cash_flow_yield":
                    (fcf / shares / px) if fcf is not None and shares and px
                    else None,
                "return_on_equity":
                    (f["net_income"] / f["equity"])
                    if f.get("net_income") is not None and f.get("equity")
                    else None,
                "gross_margin":
                    (f["gross_profit"] / rev)
                    if f.get("gross_profit") is not None and rev else None,
                "earnings_per_share_growth":
                    ((eps - peps) / abs(peps))
                    if eps is not None and peps else None,
                "revenue_growth":
                    ((rev - prev_rev) / abs(prev_rev))
                    if rev is not None and prev_rev else None,
            })
        return rows

    # ------------------------------------------------------------ filings
    def filing_events(self, ticker: str, limit: int = 500) -> list[dict]:
        """8-K events newest-first: {date, kind ('earnings'|'other'), adsh}."""
        return self._sec.filing_events(ticker, limit=limit)

    # ------------------------------------------------------------ macro (FRED)
    def _fred_client(self):
        if self._fred is None:
            from fund.data.clients.fred import FREDClient
            self._fred = FREDClient(cache_dir=self._fred_cache_dir)
        return self._fred

    def yield_curve_history(self, limit: int = 400) -> list[dict]:
        return self._fred_client().yield_curve_history(limit=limit)

    def inflation_history(self, limit: int = 48) -> list[dict]:
        return self._fred_client().inflation_history(limit=limit)

    def unemployment_history(self, limit: int = 48) -> list[dict]:
        return self._fred_client().unemployment_history(limit=limit)
