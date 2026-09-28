"""Monitoring agent (live loop; not used by backtests).

Responsibilities in the live fund:
  * daily stop-loss / thesis-invalidation checks on held positions
  * corporate-action watch
  * kill-switch: flatten on data outage, API errors, or drawdown
    circuit-breaker (e.g. -8% from equity peak pauses new entries)

Stub for now -- Phase 3, after research + decision are validated.
"""
from __future__ import annotations


class Monitor:
    def __init__(self):
        raise NotImplementedError("Monitoring agent is Phase 3.")
