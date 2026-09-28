"""Alpaca PAPER-ONLY execution wrapper for the live loop.

Raw HTTPS against https://paper-api.alpaca.markets -- deliberately no
alpaca-py dependency, and the paper endpoint is hardcoded: constructing
the broker with any other base URL raises ValueError. There is no code
path in this module that can reach the live trading endpoint.

API keys come from ALPACA_API_KEY / ALPACA_API_SECRET env vars, falling
back to the repo-root .env (gitignored). Keys are never written anywhere
else, never logged, never printed.

DryRunBroker implements the same interface with zero network traffic for
--dry-run mode and for unit tests.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Sandbox egress goes through the proxy.
for var in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.setdefault(var, "http://198.19.0.1:3128")

# THE only endpoint this module may talk to. The live URL
# (https://api.alpaca.markets) is refused explicitly -- see __init__.
PAPER_API_BASE = "https://paper-api.alpaca.markets"
_LIVE_API_BASE = "https://api.alpaca.markets"  # named only to refuse it

_TIMEOUT_S = 30


class BrokerError(Exception):
    """Anything the broker could not do (auth, network, API rejection)."""


def _read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (no quoting rules needed for our .env)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def load_api_keys() -> tuple[str, str]:
    """Return (key, secret) from env, falling back to the repo-root .env."""
    key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
    secret = (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or "").strip()
    if not key or not secret:
        env = _read_env_file(Path(__file__).resolve().parents[3] / ".env")
        key = key or env.get("ALPACA_API_KEY", "").strip() or env.get("APCA_API_KEY_ID", "").strip()
        secret = (secret or env.get("ALPACA_API_SECRET", "").strip()
                  or env.get("APCA_API_SECRET_KEY", "").strip())
    return key, secret


def alpaca_symbol(ticker: str) -> str:
    """Panel-style 'BRK-B' is already Alpaca-style; dotted 'BRK.B' -> 'BRK-B'."""
    s = (ticker or "").strip().upper()
    return s.replace(".", "-")


class PaperBroker:
    """Thin wrapper over the Alpaca paper trading REST API. Long-only."""

    def __init__(self, api_key: str = "", api_secret: str = "",
                 base_url: str = PAPER_API_BASE) -> None:
        if base_url != PAPER_API_BASE:
            raise ValueError(
                f"refusing non-paper Alpaca endpoint: {base_url!r} "
                f"(only {PAPER_API_BASE!r} is allowed)"
            )
        if _LIVE_API_BASE in base_url:
            raise ValueError("refusing live Alpaca endpoint")
        key, secret = api_key.strip(), api_secret.strip()
        if not key or not secret:
            key, secret = load_api_keys()
        if not key or not secret:
            raise BrokerError(
                "ALPACA_API_KEY / ALPACA_API_SECRET not set (env or repo .env)"
            )
        self._key = key
        self._secret = secret
        self.base_url = PAPER_API_BASE

    # ------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str,
                 data: Optional[dict] = None) -> Any:
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=body, method=method.upper())
        req.add_header("APCA-API-KEY-ID", self._key)
        req.add_header("APCA-API-SECRET-KEY", self._secret)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise BrokerError(f"{method} {path} -> HTTP {e.code}: {detail}")
        except OSError as e:
            raise BrokerError(f"{method} {path} -> network error: {e}")

    # --------------------------------------------------------------- reads
    def get_account(self) -> dict:
        """Account snapshot: status, equity, cash, buying_power, currency."""
        a = self._request("GET", "/v2/account")
        return {
            "status": a.get("status"),
            "equity": float(a.get("equity") or 0),
            "cash": float(a.get("cash") or 0),
            "buying_power": float(a.get("buying_power") or 0),
            "currency": a.get("currency"),
        }

    def get_positions(self) -> dict[str, dict]:
        """symbol -> {qty, market_value, avg_entry_price, current_price}."""
        out: dict[str, dict] = {}
        for p in self._request("GET", "/v2/positions") or []:
            sym = str(p.get("symbol", "")).upper()
            out[sym] = {
                "qty": float(p.get("qty") or 0),
                "market_value": float(p.get("market_value") or 0),
                "avg_entry_price": float(p.get("avg_entry_price") or 0),
                "current_price": float(p.get("current_price") or 0),
            }
        return out

    # -------------------------------------------------------------- orders
    def submit_order(self, symbol: str, side: str, *,
                     notional: Optional[float] = None,
                     qty: Optional[float] = None,
                     time_in_force: str = "day") -> dict:
        """Market order. Buys use notional ($); sells use qty (shares)."""
        side = side.lower()
        if side not in ("buy", "sell"):
            raise BrokerError(f"refusing order with side={side!r} (long-only)")
        if (notional is None) == (qty is None):
            raise BrokerError("specify exactly one of notional / qty")
        if notional is not None and notional <= 0:
            raise BrokerError(f"refusing non-positive notional {notional}")
        if qty is not None and qty <= 0:
            raise BrokerError(f"refusing non-positive qty {qty}")
        payload: dict[str, Any] = {
            "symbol": alpaca_symbol(symbol),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        if notional is not None:
            payload["notional"] = round(float(notional), 2)
        else:
            payload["qty"] = round(float(qty), 6)
        o = self._request("POST", "/v2/orders", payload)
        return {
            "id": o.get("id"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "notional": o.get("notional"),
            "qty": o.get("qty"),
            "filled_qty": o.get("filled_qty"),
            "filled_avg_price": o.get("filled_avg_price"),
            "status": o.get("status"),
        }

    def close_position(self, symbol: str) -> dict:
        """Liquidate one position at market."""
        o = self._request("DELETE", f"/v2/positions/{alpaca_symbol(symbol)}")
        return {"symbol": symbol, "status": o.get("status", "closed")}


@dataclass
class DryRunBroker:
    """Offline stand-in: same interface, records calls, zero HTTP traffic.

    Used for --dry-run mode and unit tests. `positions` maps symbol ->
    {qty, market_value, avg_entry_price, current_price}; `account_equity`
    stands in for the account.
    """

    positions: dict[str, dict] = field(default_factory=dict)
    account_equity: float = 100_000.0
    calls: list = field(default_factory=list)

    base_url: str = PAPER_API_BASE  # interface parity; never used

    def get_account(self) -> dict:
        self.calls.append(("get_account",))
        return {"status": "DRY-RUN", "equity": self.account_equity,
                "cash": self.account_equity, "buying_power": self.account_equity,
                "currency": "USD"}

    def get_positions(self) -> dict[str, dict]:
        self.calls.append(("get_positions",))
        return dict(self.positions)

    def submit_order(self, symbol: str, side: str, *,
                     notional: Optional[float] = None,
                     qty: Optional[float] = None,
                     time_in_force: str = "day") -> dict:
        raise BrokerError("DryRunBroker never submits orders (dry-run mode)")

    def close_position(self, symbol: str) -> dict:
        raise BrokerError("DryRunBroker never closes positions (dry-run mode)")
