"""Alpaca paper-trading execution wrapper (live loop; not used by backtests).

Reuses the client patterns from the Baskets codebase
(`fund.data.clients.alpaca`). Stub for now: the live loop is Phase 5,
after the walk-forward validation in Phase 4.
"""
from __future__ import annotations


class PaperBroker:
    def __init__(self):
        raise NotImplementedError(
            "Live execution is Phase 5. Requires Alpaca API keys in the "
            "user's .env (never stored in this repo)."
        )
