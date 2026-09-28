from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class FinnhubConfig:
    api_key: str
    base_url: str = "https://finnhub.io/api/v1"


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }


def _is_rate_limited(resp: requests.Response | None, exc: BaseException | None) -> bool:
    if resp is not None and resp.status_code in (429, 503):
        return True
    if exc is None:
        return False
    msg = str(exc).lower()
    return "rate" in msg and ("limit" in msg or "too many" in msg)


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: Dict,
    attempts: int = 3,
    base_sleep_s: float = 0.9,
) -> Dict:
    last_exc: BaseException | None = None
    last_resp: requests.Response | None = None
    for i in range(attempts):
        try:
            resp = session.get(url, params=params, timeout=20, headers=_headers())
            last_resp = resp
            if resp.status_code in (429, 503):
                raise RuntimeError(f"Finnhub rate limited ({resp.status_code})")
            if resp.status_code == 403:
                # Check if it's a forbidden error - might be API key permissions
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", "Forbidden")
                except:
                    error_msg = "Forbidden - API key may not have access to this endpoint"
                raise RuntimeError(f"Finnhub 403 Forbidden: {error_msg}. Your API key may not have access to historical candle data. Check your Finnhub subscription tier.")
            resp.raise_for_status()
            return resp.json()
        except BaseException as e:
            last_exc = e
            if i < attempts - 1 and _is_rate_limited(last_resp, last_exc):
                time.sleep(base_sleep_s * (2**i))
                continue
            raise
    raise last_exc or RuntimeError("Finnhub request failed")


def _to_unix(ts: date) -> int:
    return int(datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc).timestamp())


def _datetime_to_unix(dt: datetime) -> int:
    """Convert datetime to Unix timestamp (seconds)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_candles_5min(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    cfg: FinnhubConfig,
    session: requests.Session,
) -> Optional[Tuple[List[datetime], List[float]]]:
    """
    Returns (timestamps, closes) for 5-minute candles.
    Uses Finnhub stock/candle with resolution=5.
    """
    url = f"{cfg.base_url}/stock/candle"
    j = _get_json(
        session,
        url,
        params={
            "symbol": symbol,
            "resolution": "5",
            "from": _datetime_to_unix(start),
            "to": _datetime_to_unix(end),
            "adjusted": "true",
            "token": cfg.api_key,
        },
    )
    if j.get("s") != "ok":
        return None
    t = j.get("t") or []
    c = j.get("c") or []
    if not t or not c or len(t) != len(c):
        return None
    times = [datetime.fromtimestamp(int(x), tz=timezone.utc).replace(tzinfo=None) for x in t]
    closes = [float(x) for x in c]
    return times, closes


def fetch_daily_candles(
    *,
    symbol: str,
    start: date,
    end: date,
    cfg: FinnhubConfig,
    session: requests.Session,
) -> Optional[Tuple[List[datetime], List[float]]]:
    """
    Returns (timestamps, closes) for daily candles.
    Uses adjusted=true to account for splits (Finnhub docs: splits adjusted on D/W/M).
    """
    url = f"{cfg.base_url}/stock/candle"
    j = _get_json(
        session,
        url,
        params={
            "symbol": symbol,
            "resolution": "D",
            "from": _to_unix(start),
            "to": _to_unix(end),
            "adjusted": "true",
            "token": cfg.api_key,
        },
    )
    if j.get("s") != "ok":
        return None
    t = j.get("t") or []
    c = j.get("c") or []
    if not t or not c or len(t) != len(c):
        return None
    times = [datetime.fromtimestamp(int(x), tz=timezone.utc).replace(tzinfo=None) for x in t]
    closes = [float(x) for x in c]
    return times, closes


def fetch_dividends(
    *,
    symbol: str,
    start: date,
    end: date,
    cfg: FinnhubConfig,
    session: requests.Session,
) -> Dict[date, float]:
    """
    Return a mapping ex-date -> dividend amount per share (cash).
    """
    url = f"{cfg.base_url}/stock/dividend"
    j = _get_json(
        session,
        url,
        params={
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": cfg.api_key,
        },
    )
    out: Dict[date, float] = {}
    if not isinstance(j, list):
        return out
    for row in j:
        try:
            ex = row.get("date")
            amt = row.get("amount")
            if not ex or amt is None:
                continue
            d = datetime.strptime(ex, "%Y-%m-%d").date()
            out[d] = float(amt)
        except Exception:
            continue
    return out


def fetch_quote(
    *,
    symbol: str,
    cfg: FinnhubConfig,
    session: requests.Session,
) -> Optional[Dict[str, float]]:
    """
    Fetch real-time quote data.
    Returns dict with 'c' (current price), 'h' (high), 'l' (low), 'o' (open), 'pc' (previous close), 'v' (volume).
    """
    url = f"{cfg.base_url}/quote"
    try:
        j = _get_json(
            session,
            url,
            params={
                "symbol": symbol,
                "token": cfg.api_key,
            },
        )
        if not isinstance(j, dict):
            return None
        # Return current price and volume
        return {
            "current": j.get("c"),
            "high": j.get("h"),
            "low": j.get("l"),
            "open": j.get("o"),
            "previous_close": j.get("pc"),
            "volume": j.get("v"),
        }
    except Exception:
        return None


def fetch_company_profile(
    *,
    symbol: str,
    cfg: FinnhubConfig,
    session: requests.Session,
) -> Optional[Dict]:
    """
    Fetch company profile information.
    Returns dict with company name, sector, industry, market cap, etc.
    """
    url = f"{cfg.base_url}/stock/profile2"
    try:
        j = _get_json(
            session,
            url,
            params={
                "symbol": symbol,
                "token": cfg.api_key,
            },
        )
        if not isinstance(j, dict) or j.get("name") is None:
            return None
        return j
    except Exception:
        return None


def fetch_earnings_calendar(
    *,
    symbol: str,
    cfg: FinnhubConfig,
    session: requests.Session,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Optional[date]:
    """
    Fetch next earnings date for a symbol.
    Returns the next earnings date if available, None otherwise.
    """
    url = f"{cfg.base_url}/calendar/earnings"
    params: Dict = {
        "symbol": symbol,
        "token": cfg.api_key,
    }
    if start:
        params["from"] = start.isoformat()
    if end:
        params["to"] = end.isoformat()
    
    try:
        j = _get_json(session, url, params=params)
        if not isinstance(j, dict):
            return None
        earnings_list = j.get("earningsCalendar")
        if not isinstance(earnings_list, list) or not earnings_list:
            return None
        
        # Find the next earnings date
        today = date.today()
        for item in earnings_list:
            if not isinstance(item, dict):
                continue
            date_str = item.get("date")
            if not date_str:
                continue
            try:
                earnings_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                if earnings_date >= today:
                    return earnings_date
            except Exception:
                continue
        return None
    except Exception:
        return None


_SESSION: requests.Session | None = None


def get_finnhub_session() -> requests.Session:
    """Return a shared requests Session for Finnhub API calls."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(_headers())
    return _SESSION


def get_finnhub_config() -> Optional[FinnhubConfig]:
    """
    Get Finnhub config from environment.
    Loads from backend/.env if FINNHUB_API_KEY not already set.
    """
    api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        # Best-effort: load backend/.env
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
        api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        return None
    return FinnhubConfig(api_key=api_key)

