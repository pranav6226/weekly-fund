"""Financial Datasets API client -- the fund's multi-source data stack.

Stolen from virattt/dexter (src/tools/finance/): one provider for prices,
fundamentals, earnings, news, and macro, with the endpoint paths Dexter's
27k-star codebase settled on:
  GET /financial-metrics/   historical metrics (carries filing_date --
                            the field that makes point-in-time backtests
                            possible)
  GET /earnings             structured results + surprise flags
  GET /news                 headlines with publication dates
  GET /macro/...            treasury yields, inflation, labor

Caching steals Dexter's TTL idea: fundamentals are slow-moving (24h TTL),
news is fast (1h TTL). Cache lives in data/fd_cache/ as JSON keyed by
endpoint+params hash -- no secrets ever touch the cache.

Auth: FINANCIAL_DATASETS_API_KEY env var (his .env). Nothing is stored here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

BASE_URL = "https://api.financialdatasets.ai"

# Dexter's TTLs, adopted: fundamentals barely move day to day, news does.
TTL_FUNDAMENTALS = 24 * 3600
TTL_NEWS = 1 * 3600


class FinancialDatasetsError(RuntimeError):
    pass


class FinancialDatasetsClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = "data/fd_cache",
    ):
        key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if not key:
            raise FinancialDatasetsError(
                "No Financial Datasets API key. Set FINANCIAL_DATASETS_API_KEY "
                "in the environment (his .env). Free keys at financialdatasets.ai; "
                "AAPL/MSFT/NVDA/TSLA/GOOGL data is free."
            )
        self._key = key
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"X-API-KEY": key})

    # ------------------------------------------------------------ plumbing
    def _cache_path(self, endpoint: str, params: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps({"e": endpoint, "p": params}, sort_keys=True).encode()
        ).hexdigest()[:16]
        safe = endpoint.strip("/").replace("/", "_") or "root"
        return self._cache / f"{safe}_{digest}.json"

    def _get(self, endpoint: str, params: dict | None = None,
             ttl: int = TTL_FUNDAMENTALS) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cp = self._cache_path(endpoint, params)
        if cp.exists():
            try:
                blob = json.loads(cp.read_text())
                if time.time() - blob["ts"] < ttl:
                    return blob["data"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt cache entry: refetch
        url = f"{BASE_URL}{endpoint}"
        try:
            r = self._session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            raise FinancialDatasetsError(f"FD request failed {endpoint}: {e}")
        if r.status_code == 401:
            raise FinancialDatasetsError("FD 401: bad or missing API key")
        if r.status_code == 429:
            raise FinancialDatasetsError("FD 429: rate limited -- back off")
        if r.status_code >= 400:
            raise FinancialDatasetsError(
                f"FD {r.status_code} on {endpoint}: {r.text[:200]}")
        try:
            data = r.json()
        except json.JSONDecodeError:
            raise FinancialDatasetsError(f"FD non-JSON response on {endpoint}")
        cp.write_text(json.dumps({"ts": time.time(), "data": data}))
        return data

    # ------------------------------------------------------------ datasets
    def metrics(self, ticker: str, period: str = "quarterly",
                limit: int = 12) -> list[dict]:
        """Historical financial metrics, newest first.

        Each row carries filing_date / filing_datetime -- the desks filter
        `filing_date <= as_of` for point-in-time discipline.
        """
        data = self._get("/financial-metrics/",
                         {"ticker": ticker.upper(), "period": period,
                          "limit": limit})
        return data.get("financial_metrics", [])

    def earnings(self, ticker: str, limit: int = 8) -> list[dict]:
        """Structured earnings: actuals, estimates, surprise flags."""
        data = self._get("/earnings",
                         {"ticker": ticker.upper(), "limit": limit})
        return data.get("earnings", [])

    def news(self, ticker: str, limit: int = 10) -> list[dict]:
        """Headlines with publication dates. TTL 1h -- news decays fast."""
        data = self._get("/news",
                         {"ticker": ticker.upper(), "limit": min(limit, 10)},
                         ttl=TTL_NEWS)
        return data.get("news", [])

    def market_news(self, limit: int = 10) -> list[dict]:
        """Broad market news (macro, rates, geopolitics) -- no ticker."""
        data = self._get("/news", {"limit": min(limit, 10)}, ttl=TTL_NEWS)
        return data.get("news", [])

    # --- macro (endpoint paths per docs.financialdatasets.ai; fallbacks
    # --- tried in order because historical-vs-snapshot naming varies)
    def _macro_first(self, paths: list[str], params: dict | None = None):
        last: Exception | None = None
        for p in paths:
            try:
                return self._get(p, params)
            except FinancialDatasetsError as e:
                last = e
        raise FinancialDatasetsError(f"all macro paths failed: {last}")

    def yield_curve_history(self, limit: int = 60) -> list[dict]:
        return self._macro_first(
            ["/macro/yield-curve/historical", "/macro/yield-curve/"],
            {"limit": limit}).get("yield_curves", [])

    def inflation_history(self, limit: int = 36) -> list[dict]:
        data = self._macro_first(
            ["/macro/inflation/historical", "/macro/inflation/"],
            {"limit": limit})
        return data.get("inflation", data.get("cpi", []))
