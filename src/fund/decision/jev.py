"""TypeSafe Jev interface (System One decision model).

Jev is the confidence gate for the live fund: it takes research-agent
outputs and produces calibrated position convictions via its
Score/Choice/Noul judgment primitives.

NOT wired yet, by design:
  * the API key lives in Pranav's .env and is never stored here
  * backtests must be deterministic and reproducible, so the pipeline uses
    the local fusion in fund.decision.fusion (fuse_scores) instead

When we wire it, `JevClient.convictions()` returns {ticker: conviction}
with the SAME contract as fuse_scores() -- list[AgentScore] in,
{ticker: 0..1} out -- so the simulator and portfolio constructor consume
it with zero code changes. Just swap the call in simulator.py.
"""
from __future__ import annotations

from fund.research.schema import AgentScore


class JevClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def convictions(self, scores: list[AgentScore]) -> dict[str, float]:
        raise NotImplementedError(
            "Jev wiring is for the live loop (key in user .env). "
            "Backtests use fund.research.agents.score_universe()."
        )
