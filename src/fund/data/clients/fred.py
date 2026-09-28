"""FRED client -- free macro data (yields, inflation, labor).

The key is free: register at fred.stlouisfed.org (no payment, ~1 minute)
and set FRED_API_KEY in the .env. One key unlocks the whole macro desk.
Fails loudly without it -- the macro desk must never silently run on
nothing.

Shapes match what the macro desk expects:
  yield_curve_history(limit) -> [{date, 10_year, 2_year}] newest first
  inflation_history(limit)   -> [{date, yoy_change}] newest first (CPI YoY %)
  unemployment_history(limit)-> [{date, rate}] newest first
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
TTL_MACRO = 24 * 3600


class FREDError(RuntimeError):
    pass


class FREDClient:
    def __init__(self, api_key: str | None = None,
                 cache_dir: str | Path = "data/fred_cache"):
        from fund.data.clients.cache import TTLCache
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise FREDError(
                "No FRED API key. The key is FREE -- register at "
                "https://fred.stlouisfed.org (no payment), then set "
                "FRED_API_KEY in the .env. The macro desk stays dark "
                "until then; nothing else is affected."
            )
        self._key = key
        self._cache = TTLCache(cache_dir)

    def _series(self, series_id: str) -> list[dict]:
        ns = f"fred_{series_id}"
        hit = self._cache.get(ns, {}, TTL_MACRO)
        if hit is not None:
            return hit
        try:
            r = requests.get(BASE_URL, params={
                "series_id": series_id, "api_key": self._key,
                "file_type": "json",
            }, timeout=30)
            r.raise_for_status()
            obs = [{"date": o["date"], "value": o["value"]}
                   for o in r.json().get("observations", [])
                   if o["value"] != "."]
        except requests.RequestException as e:
            raise FREDError(f"FRED {series_id} failed: {e}") from e
        self._cache.put(ns, {}, obs)
        return obs

    def yield_curve_history(self, limit: int = 400) -> list[dict]:
        d10 = {o["date"]: float(o["value"]) for o in self._series("DGS10")}
        d2 = {o["date"]: float(o["value"]) for o in self._series("DGS2")}
        dates = sorted(set(d10) & set(d2), reverse=True)[:limit]
        return [{"date": d, "10_year": d10[d], "2_year": d2[d]} for d in dates]

    def inflation_history(self, limit: int = 48) -> list[dict]:
        cpi = [(o["date"], float(o["value"]))
               for o in self._series("CPIAUCSL")]
        out = []
        for i in range(11, len(cpi)):
            d, v = cpi[i]
            _, v0 = cpi[i - 11]  # ~12 months back (monthly series)
            out.append({"date": d, "yoy_change": 100 * (v / v0 - 1)})
        out.sort(key=lambda r: r["date"], reverse=True)
        return out[:limit]

    def unemployment_history(self, limit: int = 48) -> list[dict]:
        obs = self._series("UNRATE")
        out = [{"date": o["date"], "rate": float(o["value"])} for o in obs]
        out.sort(key=lambda r: r["date"], reverse=True)
        return out[:limit]
