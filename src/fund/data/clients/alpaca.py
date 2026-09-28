from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Rate limit: min delay between Alpaca API calls (seconds). Free tier ~200/min → ~0.35s between calls.
_ALPACA_MIN_DELAY_S = float(os.getenv("ALPACA_RATE_LIMIT_DELAY_S", "0.4"))
_ALPACA_LAST_CALL = 0.0
_ALPACA_LOCK = threading.Lock()


def _wait_alpaca_rate_limit() -> None:
    """Ensure we don't exceed Alpaca rate limits by waiting if the last call was too recent."""
    if _ALPACA_MIN_DELAY_S <= 0:
        return
    with _ALPACA_LOCK:
        global _ALPACA_LAST_CALL
        now = time.monotonic()
        wait = _ALPACA_LAST_CALL + _ALPACA_MIN_DELAY_S - now
        if wait > 0:
            time.sleep(wait)
        _ALPACA_LAST_CALL = time.monotonic()


def _is_rate_limit_error(e: Exception) -> bool:
    msg = (getattr(e, "message", None) or str(e)).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg


def normalize_symbol_for_api(symbol: str) -> str:
    """Convert ticker to Alpaca format. E.g. MOG.A (invalid) -> MOG-A (Class A)."""
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if s.endswith(".A"):
        return s[:-2] + "-A"
    if s.endswith(".B"):
        return s[:-2] + "-B"
    if s.endswith(".C"):
        return s[:-2] + "-C"
    return s


def _symbol_variants_for_alpaca(symbol: str) -> List[str]:
    """Return symbol variants to try (Alpaca may accept MOG.A or MOG-A for class shares)."""
    s = (symbol or "").strip().upper()
    if not s:
        return []
    norm = normalize_symbol_for_api(s)
    variants: List[str] = []
    for candidate in (norm, s):
        if candidate and candidate not in variants:
            variants.append(candidate)
    # If we already have a hyphenated class-share symbol, also try the dotted form.
    if "-" in norm:
        dotted = re.sub(r"-(?=[A-Z]$)", ".", norm)
        if dotted and dotted not in variants:
            variants.append(dotted)
    return variants


_INVALID_SYMBOL_RE = re.compile(r"invalid symbol:\s*([A-Z0-9._-]+)", re.IGNORECASE)


def _extract_invalid_symbol(exc: Exception) -> str | None:
    match = _INVALID_SYMBOL_RE.search(str(exc))
    if not match:
        return None
    return match.group(1).strip().upper() or None


from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestBarRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://data.alpaca.markets"


def get_alpaca_config() -> Optional[AlpacaConfig]:
    """
    Get Alpaca config from environment.
    Loads from backend/.env if ALPACA_API_KEY not already set.
    """
    api_key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
    api_secret = (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or "").strip()
    
    if not api_key or not api_secret:
        # Best-effort: load backend/.env
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
        api_key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
        api_secret = (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or "").strip()
    
    if not api_key or not api_secret:
        return None
    
    return AlpacaConfig(api_key=api_key, api_secret=api_secret)


_CLIENT: StockHistoricalDataClient | None = None
_NEWS_CLIENT: NewsClient | None = None


def get_alpaca_client() -> Optional[StockHistoricalDataClient]:
    """Return a shared Alpaca client instance."""
    global _CLIENT
    cfg = get_alpaca_config()
    if not cfg:
        return None
    
    if _CLIENT is None:
        _CLIENT = StockHistoricalDataClient(
            api_key=cfg.api_key,
            secret_key=cfg.api_secret,
        )
    return _CLIENT


def get_alpaca_news_client() -> Optional[NewsClient]:
    """Return a shared Alpaca News client instance."""
    global _NEWS_CLIENT
    cfg = get_alpaca_config()
    if not cfg:
        return None
    
    if _NEWS_CLIENT is None:
        _NEWS_CLIENT = NewsClient(
            api_key=cfg.api_key,
            secret_key=cfg.api_secret,
        )
    return _NEWS_CLIENT


_TRADING_CLIENT: "TradingClient | None" = None


def get_alpaca_trading_client() -> Optional["TradingClient"]:
    """Return a shared Alpaca Trading client (for assets, orders). Used e.g. for asset.tradable check."""
    global _TRADING_CLIENT
    cfg = get_alpaca_config()
    if not cfg:
        return None
    if _TRADING_CLIENT is None:
        try:
            from alpaca.trading.client import TradingClient
            _TRADING_CLIENT = TradingClient(
                api_key=cfg.api_key,
                secret_key=cfg.api_secret,
            )
        except Exception:
            return None
    return _TRADING_CLIENT


def fetch_tradable_by_symbols(symbols: List[str]) -> Dict[str, bool]:
    """
    For each symbol, fetch Alpaca asset and return tradable status.
    Returns dict symbol -> True if tradable, False if not or on error. Missing symbols are treated as not tradable.
    """
    out: Dict[str, bool] = {s: False for s in symbols}
    client = get_alpaca_trading_client()
    if not client:
        return out
    for sym in symbols:
        _wait_alpaca_rate_limit()
        try:
            norm = normalize_symbol_for_api(sym)
            asset = client.get_asset(norm)
            if asset is not None and getattr(asset, "tradable", False):
                out[sym] = True
            # If normalized symbol failed, try original
            if not out[sym] and norm != sym:
                asset = client.get_asset(sym)
                if asset is not None and getattr(asset, "tradable", False):
                    out[sym] = True
        except Exception:
            pass
    return out


def fetch_daily_candles(
    *,
    symbol: str,
    start: date,
    end: date,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Optional[Tuple[List[datetime], List[float]]]:
    """
    Returns (timestamps, closes) for daily candles.
    Uses Alpaca's historical bars API with daily timeframe.
    Data is automatically adjusted for splits and dividends.
    Tries normalized symbol (e.g. MOG-A) first, then original (e.g. MOG.A) if Alpaca rejects it.
    """
    client = session or get_alpaca_client()
    if not client:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET not configured")
    variants = _symbol_variants_for_alpaca(symbol)
    last_exc = None
    for sym in variants:
        _wait_alpaca_rate_limit()
        try:
            start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc)
            request_params = StockBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment="all",
                feed="iex",
            )
            bars = client.get_stock_bars(request_params)
            if not bars:
                continue
            try:
                symbol_bars = bars[sym]
            except (KeyError, TypeError):
                if hasattr(bars, "data") and isinstance(bars.data, dict):
                    symbol_bars = bars.data.get(sym)
                else:
                    continue
            if not symbol_bars or len(symbol_bars) == 0:
                continue
            times = [bar.timestamp.replace(tzinfo=None) for bar in symbol_bars]
            closes = [float(bar.close) for bar in symbol_bars]
            return times, closes
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e):
                time.sleep(2.0)
            continue
    if last_exc:
        import logging
        logging.warning(f"Alpaca fetch_daily_candles failed for {symbol}: {last_exc}")
    return None


def fetch_minute_candles(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Optional[Tuple[List[datetime], List[float]]]:
    """
    Returns (timestamps, closes) for minute candles.
    Uses Alpaca historical bars API with minute timeframe on IEX feed.
    Tries normalized symbol first, then original (e.g. MOG.A) if Alpaca rejects MOG-A.
    """
    client = session or get_alpaca_client()
    if not client:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET not configured")
    orig = (symbol or "").strip().upper()
    variants = _symbol_variants_for_alpaca(symbol)
    for sym in variants:
        _wait_alpaca_rate_limit()
        try:
            start_dt = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            end_dt = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            request_params = StockBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=TimeFrame.Minute,
                start=start_dt,
                end=end_dt,
                adjustment="all",
                feed="iex",
            )
            bars = client.get_stock_bars(request_params)
            if not bars:
                continue
            try:
                symbol_bars = bars[sym]
            except (KeyError, TypeError):
                if hasattr(bars, "data") and isinstance(bars.data, dict):
                    symbol_bars = bars.data.get(sym)
                else:
                    continue
            if not symbol_bars or len(symbol_bars) == 0:
                continue
            times = [bar.timestamp.replace(tzinfo=None) for bar in symbol_bars]
            closes = [float(bar.close) for bar in symbol_bars]
            return times, closes
        except Exception as e:
            import logging
            logging.warning(f"Alpaca fetch_minute_candles failed for {orig} (tried {sym}): {e}")
            continue
    return None


def fetch_minute_candles_batch(
    *,
    symbols: List[str],
    start: datetime,
    end: datetime,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Dict[str, Tuple[List[datetime], List[float]]]:
    """
    Fetch minute candles for multiple symbols with a single Alpaca request.
    Returns: symbol -> (timestamps, closes). Tries original (e.g. MOG.A) for class shares if normalized fails.
    """
    client = session or get_alpaca_client()
    if not client:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET not configured")

    uniq: List[str] = []
    seen = set()
    for s in symbols or []:
        t = str(s or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return {}
    norm_uniq = [normalize_symbol_for_api(t) for t in uniq]
    _wait_alpaca_rate_limit()
    out: Dict[str, Tuple[List[datetime], List[float]]] = {}
    try:
        start_dt = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_dt = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        request_params = StockBarsRequest(
            symbol_or_symbols=norm_uniq,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
            adjustment="all",
            feed="iex",
        )
        bars = client.get_stock_bars(request_params)
        if bars:
            data = bars.data if hasattr(bars, "data") and isinstance(bars.data, dict) else (bars if isinstance(bars, dict) else {})
            for orig, norm in zip(uniq, norm_uniq):
                sym_bars = data.get(norm)
                if not sym_bars:
                    continue
                times = [bar.timestamp.replace(tzinfo=None) for bar in sym_bars]
                closes = [float(bar.close) for bar in sym_bars]
                if times and closes and len(times) == len(closes):
                    out[orig] = (times, closes)
    except Exception as e:
        import logging
        logging.warning(f"Alpaca fetch_minute_candles_batch failed for {len(uniq)} symbols: {e}")

    # For any symbol that failed (including full batch failures), try per-symbol variants.
    missing = [orig for orig in uniq if orig not in out]
    for orig in missing:
        variants = _symbol_variants_for_alpaca(orig)
        for alt in variants:
            _wait_alpaca_rate_limit()
            try:
                start_dt = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
                end_dt = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
                request_params = StockBarsRequest(
                    symbol_or_symbols=[alt],
                    timeframe=TimeFrame.Minute,
                    start=start_dt,
                    end=end_dt,
                    adjustment="all",
                    feed="iex",
                )
                bars = client.get_stock_bars(request_params)
                if not bars:
                    continue
                data = bars.data if hasattr(bars, "data") and isinstance(bars.data, dict) else (bars if isinstance(bars, dict) else {})
                sym_bars = data.get(alt)
                if not sym_bars:
                    continue
                times = [bar.timestamp.replace(tzinfo=None) for bar in sym_bars]
                closes = [float(bar.close) for bar in sym_bars]
                if times and closes and len(times) == len(closes):
                    out[orig] = (times, closes)
                    break
            except Exception:
                continue
    return out


def fetch_dividends(
    *,
    symbol: str,
    start: date,
    end: date,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Dict[date, float]:
    """
    Return a mapping ex-date -> dividend amount per share (cash).
    Note: Alpaca doesn't have a direct dividends endpoint in the free tier.
    This is a placeholder that returns empty dict.
    For dividend data, you may need to use a different service or upgrade.
    """
    # Alpaca's free tier doesn't provide dividend data via API
    # The bars data is already adjusted for dividends, so we return empty
    # If you need dividend data, consider using a different provider or upgrading
    return {}


def fetch_quote(
    *,
    symbol: str,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Optional[Dict[str, float]]:
    """
    Fetch real-time quote data.
    Returns dict with 'current' (current price), 'high', 'low', 'open', 'previous_close', 'volume'.
    Uses latest bar data for OHLC and latest quote for current price.
    """
    symbol = normalize_symbol_for_api(symbol)
    client = session or get_alpaca_client()
    if not client:
        return None
    _wait_alpaca_rate_limit()
    try:
        # Get latest bar for OHLC data (use IEX feed - free, no SIP subscription required)
        bar_request = StockLatestBarRequest(symbol_or_symbols=[symbol], feed="iex")
        bars = client.get_stock_latest_bar(bar_request)
        
        # Get latest quote for current bid/ask
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed="iex")
        quotes = client.get_stock_latest_quote(quote_request)
        
        current_price = None
        if quotes and symbol in quotes and quotes[symbol]:
            quote = quotes[symbol]
            # Use midpoint of bid/ask as current price
            if quote.bid_price and quote.ask_price:
                current_price = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
            elif quote.ask_price:
                current_price = float(quote.ask_price)
            elif quote.bid_price:
                current_price = float(quote.bid_price)
        
        if bars and symbol in bars and bars[symbol]:
            bar = bars[symbol]
            return {
                "current": current_price or float(bar.close) if bar.close else None,
                "high": float(bar.high) if bar.high else None,
                "low": float(bar.low) if bar.low else None,
                "open": float(bar.open) if bar.open else None,
                "previous_close": float(bar.close) if bar.close else None,
                "volume": int(bar.volume) if bar.volume else None,
            }
        elif current_price is not None:
            # If we have quote but no bar, return just the current price
            return {
                "current": current_price,
                "high": None,
                "low": None,
                "open": None,
                "previous_close": None,
                "volume": None,
            }
        
        return None
    except Exception as e:
        import logging
        if _is_rate_limit_error(e):
            time.sleep(2.0)
            _wait_alpaca_rate_limit()
            try:
                bar_request = StockLatestBarRequest(symbol_or_symbols=[symbol], feed="iex")
                bars = client.get_stock_latest_bar(bar_request)
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed="iex")
                quotes = client.get_stock_latest_quote(quote_request)
                current_price = None
                if quotes and symbol in quotes and quotes[symbol]:
                    quote = quotes[symbol]
                    if quote.bid_price and quote.ask_price:
                        current_price = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
                    elif quote.ask_price:
                        current_price = float(quote.ask_price)
                    elif quote.bid_price:
                        current_price = float(quote.bid_price)
                if bars and symbol in bars and bars[symbol]:
                    bar = bars[symbol]
                    return {
                        "current": current_price or float(bar.close) if bar.close else None,
                        "high": float(bar.high) if bar.high else None,
                        "low": float(bar.low) if bar.low else None,
                        "open": float(bar.open) if bar.open else None,
                        "previous_close": float(bar.close) if bar.close else None,
                        "volume": int(bar.volume) if bar.volume else None,
                    }
                elif current_price is not None:
                    return {
                        "current": current_price,
                        "high": None,
                        "low": None,
                        "open": None,
                        "previous_close": None,
                        "volume": None,
                    }
                return None
            except Exception:
                pass
        logging.warning(f"Alpaca fetch_quote failed for {symbol}: {e}")
        return None


def fetch_latest_bars(
    *,
    symbols: List[str],
    session: StockHistoricalDataClient | None = None,
) -> Dict[str, Dict[str, float | int | None]]:
    """
    Fetch latest bar data for a list of symbols.
    Returns dict: symbol -> {open, high, low, close, volume}.
    """
    client = session or get_alpaca_client()
    if not client:
        return {}
    
    # Normalize symbols: unique, upper-case, non-empty; and to API format (e.g. MOG.A -> MOG-A)
    uniq: List[str] = []
    seen = set()
    for s in symbols or []:
        t = str(s or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return {}
    def _coerce_latest_bar_payload(bars_obj) -> Dict[str, object]:
        if hasattr(bars_obj, "data") and isinstance(bars_obj.data, dict):
            return bars_obj.data
        if isinstance(bars_obj, dict):
            return bars_obj
        return {}

    def _serialize_bar(bar) -> Dict[str, float | int | None]:
        return {
            "open": float(bar.open) if bar.open is not None else None,
            "high": float(bar.high) if bar.high is not None else None,
            "low": float(bar.low) if bar.low is not None else None,
            "close": float(bar.close) if bar.close is not None else None,
            "volume": int(bar.volume) if bar.volume is not None else None,
        }

    out: Dict[str, Dict[str, float | int | None]] = {}
    remaining_pairs = [(orig, normalize_symbol_for_api(orig)) for orig in uniq]
    invalid_symbols: List[str] = []

    while remaining_pairs:
        _wait_alpaca_rate_limit()
        try:
            request_symbols = [norm for _, norm in remaining_pairs]
            bar_request = StockLatestBarRequest(symbol_or_symbols=request_symbols, feed="iex")
            bars = client.get_stock_latest_bar(bar_request)
            if not bars:
                break

            data = _coerce_latest_bar_payload(bars)
            for orig, norm in remaining_pairs:
                bar = data.get(norm)
                if not bar:
                    continue
                out[orig] = _serialize_bar(bar)
            break
        except Exception as e:
            invalid_symbol = _extract_invalid_symbol(e)
            if invalid_symbol:
                next_remaining = [
                    (orig, norm)
                    for orig, norm in remaining_pairs
                    if invalid_symbol not in {orig.upper(), norm.upper()}
                ]
                if len(next_remaining) != len(remaining_pairs):
                    invalid_symbols.append(invalid_symbol)
                    remaining_pairs = next_remaining
                    continue
            import logging
            logging.warning(f"Alpaca fetch_latest_bars failed: {e}")
            break

    if invalid_symbols:
        import logging
        logging.info("Skipping invalid Alpaca latest-bar symbols: %s", ", ".join(sorted(set(invalid_symbols))))

    # Degrade gracefully: one bad symbol should not zero-out the whole chunk.
    for orig in uniq:
        if orig in out:
            continue
        variants = _symbol_variants_for_alpaca(orig)
        for sym in variants:
            _wait_alpaca_rate_limit()
            try:
                req = StockLatestBarRequest(symbol_or_symbols=[sym], feed="iex")
                bars_single = client.get_stock_latest_bar(req)
                data_single = _coerce_latest_bar_payload(bars_single)
                bar = data_single.get(sym)
                if not bar:
                    continue
                out[orig] = _serialize_bar(bar)
                break
            except Exception:
                continue
    return out


def fetch_company_profile(
    *,
    symbol: str,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
) -> Optional[Dict]:
    """
    Fetch company profile information.
    Note: Alpaca doesn't provide company profile data in their Market Data API.
    This is a placeholder that returns None.
    For company profiles, you may need to use a different service.
    """
    # Alpaca Market Data API doesn't include company profiles
    # Consider using a different provider for this data
    return None


def fetch_earnings_calendar(
    *,
    symbol: str,
    cfg: AlpacaConfig | None = None,
    session: StockHistoricalDataClient | None = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Optional[date]:
    """
    Fetch next earnings date for a symbol.
    Note: Alpaca doesn't provide earnings calendar data in their Market Data API.
    This is a placeholder that returns None.
    """
    # Alpaca Market Data API doesn't include earnings calendar
    # Consider using a different provider for this data
    return None


def fetch_company_news(
    *,
    symbol: str,
    start: date,
    end: date,
    cfg: AlpacaConfig | None = None,
    session: NewsClient | None = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Fetch company news from Alpaca.
    Returns a list of news items with: headline, datetime, url, source, summary.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date for news
        end: End date for news
        cfg: AlpacaConfig (optional, will be fetched if not provided)
        session: NewsClient (optional, will be created if not provided)
        limit: Maximum number of news items to return (default: 10)
    
    Returns:
        List of news dictionaries with keys: headline, datetime, url, source, summary
    """
    client = session or get_alpaca_news_client()
    if not client:
        return []
    _wait_alpaca_rate_limit()
    try:
        # Convert dates to datetime for Alpaca API
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=timezone.utc)
        
        request_params = NewsRequest(
            symbols=symbol,  # NewsRequest expects a string, not a list
            start=start_dt,
            end=end_dt,
            limit=limit,
        )
        
        news_set = client.get_news(request_params)
        
        if not news_set or not hasattr(news_set, 'data'):
            return []
        
        # NewsSet.data is a dict with key 'news' containing a list of News objects
        news_list = news_set.data.get('news', [])
        if not news_list:
            return []
        
        # Convert Alpaca news format to match Finnhub format
        out: List[Dict] = []
        for item in news_list[:limit]:
            if not item:
                continue
            
            # Alpaca news items have: id, headline, author, created_at, updated_at, url, content, images, symbols, source, summary
            headline = getattr(item, 'headline', None)
            created_at = getattr(item, 'created_at', None)
            url = getattr(item, 'url', None)
            content = getattr(item, 'content', None) or getattr(item, 'summary', None)
            author = getattr(item, 'author', None) or getattr(item, 'source', None)
            
            # Convert datetime to timestamp if needed (Finnhub uses milliseconds timestamp)
            datetime_val = None
            if created_at:
                if isinstance(created_at, datetime):
                    datetime_val = int(created_at.timestamp() * 1000)  # Convert to milliseconds timestamp
                elif isinstance(created_at, (int, float)):
                    datetime_val = int(created_at)
            
            out.append({
                "headline": headline,
                "datetime": datetime_val,
                "url": url,
                "source": author,
                "summary": content,
            })
        
        return out
    except Exception as e:
        # Log error but return empty list to match finnhub_client interface
        import logging
        logging.warning(f"Alpaca fetch_company_news failed for {symbol}: {e}")
        return []


def fetch_market_news(
    *,
    start: datetime,
    end: datetime,
    symbols: Optional[List[str]] = None,
    limit: int = 50,
    session: NewsClient | None = None,
) -> List[Dict]:
    """
    Fetch recent market news from Alpaca.
    Returns a list of news items with: id, headline, summary, author, created_at, updated_at, url, symbols, source.
    """
    client = session or get_alpaca_news_client()
    if not client:
        return []
    _wait_alpaca_rate_limit()
    try:
        symbols_param: Optional[str] = None
        if symbols:
            symbols_param = symbols[0] if len(symbols) == 1 else ",".join(symbols)

        request_params = NewsRequest(
            symbols=symbols_param,
            start=start,
            end=end,
            limit=limit,
        )
        news_set = client.get_news(request_params)
        if not news_set or not hasattr(news_set, "data"):
            return []

        news_list = news_set.data.get("news", [])
        if not news_list:
            return []

        out: List[Dict] = []
        for item in news_list[:limit]:
            if not item:
                continue

            out.append(
                {
                    "id": getattr(item, "id", None),
                    "headline": getattr(item, "headline", None),
                    "summary": getattr(item, "summary", None) or getattr(item, "content", None),
                    "author": getattr(item, "author", None),
                    "created_at": getattr(item, "created_at", None),
                    "updated_at": getattr(item, "updated_at", None),
                    "url": getattr(item, "url", None),
                    "symbols": getattr(item, "symbols", None) or [],
                    "source": getattr(item, "source", None),
                }
            )

        return out
    except Exception as e:
        import logging
        logging.warning(f"Alpaca fetch_market_news failed: {e}")
        return []
