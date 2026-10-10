"""GDELT news-tone client (sentiment desk data source).

Uses the GDELT DOC 2.1 API (free, no key) in `tonechart` mode: for a company
query over a date range it returns article counts bucketed by tone (-10..+10),
from which we derive average tone and coverage. Raw HTTP via urllib (the
`gdeltdoc` wrapper misclassified proxy responses as rate limits); the sandbox
egress proxy is honored from the environment.

Point-in-time discipline: every query carries an explicit [start, end] window
and only articles published inside it are counted. Tone is scored at ingest
and never revised, so historical windows are safe as long as callers filter on
publication date -- which this client enforces by construction (the window is
the query).

Responses are cached on disk keyed by (query, start, end); the cache is the
point-in-time record, so re-runs and backtests never re-hit the API.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT_SECS = 30
_MAX_RETRIES = 6
_PACE_SECS = 5.0  # min gap between requests; GDELT 429s hard through this proxy


def _proxy_opener() -> urllib.request.OpenerDirector:
    proxies = {}
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        if os.environ.get(var):
            proxies["https" if "https" in var.lower() else "http"] = os.environ[var]
            break
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


class GDELTError(Exception):
    """GDELT request failed after retries."""


class GDELTClient:
    def __init__(self, cache_dir: str | Path = "data/gdelt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._opener = _proxy_opener()
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0}
        self._last_req = 0.0

    @staticmethod
    def _cache_key(query: str, start: str, end: str) -> str:
        h = hashlib.sha256(f"{query}|{start}|{end}".encode()).hexdigest()[:16]
        return f"{h}.json"

    def _get(self, params: dict) -> dict:
        url = API_BASE + "?" + urllib.parse.urlencode(params)
        delay = 5.0
        for attempt in range(_MAX_RETRIES + 1):
            gap = time.time() - self._last_req
            if gap < _PACE_SECS:
                time.sleep(_PACE_SECS - gap)
            self._last_req = time.time()
            self.stats["requests"] += 1
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "weekly-fund/1.0",
                         "Connection": "close"},
            )
            try:
                with self._opener.open(req, timeout=_TIMEOUT_SECS) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < _MAX_RETRIES:
                    self.stats["retries"] += 1
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GDELTError(f"GDELT HTTP {e.code}") from e
            except Exception as e:  # noqa: BLE001 -- proxy flakes, retry
                if attempt < _MAX_RETRIES:
                    self.stats["retries"] += 1
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GDELTError(f"GDELT request failed: {e}") from e
        raise GDELTError("GDELT request failed after retries")  # unreachable

    def tone(self, query: str, start_date: str, end_date: str) -> dict:
        """Average news tone for `query` over [start_date, end_date].

        Dates are YYYYMMDD. Returns {"avg_tone": float|None,
        "n_articles": int, "bins": {-10: n, ...}}. avg_tone is None when no
        articles matched (caller treats as abstain, not neutral).
        """
        key = self._cache_key(query, start_date, end_date)
        p = self.cache_dir / key
        if p.exists():
            self.stats["cache_hits"] += 1
            return json.loads(p.read_text())

        params = {
            "query": query,
            "mode": "tonechart",
            "format": "json",
            "startdatetime": start_date + "000000",
            "enddatetime": end_date + "235959",
        }
        d = self._get(params)
        bins: dict[int, int] = {}
        for row in d.get("tonechart", []):
            try:
                bins[int(row["bin"])] = int(row["count"])
            except (KeyError, TypeError, ValueError):
                continue
        n = sum(bins.values())
        avg = (sum(b * c for b, c in bins.items()) / n) if n else None
        out = {"avg_tone": avg, "n_articles": n,
               "bins": {str(b): c for b, c in bins.items()}}
        p.write_text(json.dumps(out))
        return out
