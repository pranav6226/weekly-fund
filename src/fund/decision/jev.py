"""TypeSafe Jev interface (System One decision model).

Jev is the confidence gate for the live fund: it takes research-agent
outputs and produces calibrated position convictions via its
Score/Choice/Noul judgment primitives.

NOT wired yet, by design:
  * the API key lives in Pranav's .env and is never stored here
  * backtests must be deterministic and reproducible, so the pipeline uses
    the local fusion in fund.decision.fusion (fuse_desks) instead

When we wire it, `JevClient.convictions()` returns {ticker: conviction}
with the SAME contract as fuse_desks() -- list[AgentScore] in,
{ticker: 0..1} out -- so the simulator and portfolio constructor consume
it with zero code changes. Just swap the call in simulator.py.

BACKTEST RULE (stolen from ai-hedge-fund): a judge trained after the
backtest window may remember how a named company did, and that memory
scores as skill. So backtests MUST call convictions(..., blind=True):
tickers become T-0001 labels, dates are withheld, and the returned
convictions are mapped back before portfolio construction. Live runs use
blind=False and name the company.
"""
from __future__ import annotations

from fund.decision.fusion import anonymize_scores, deanonymize_convictions
from fund.research.schema import AgentScore


class JevClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def convictions(self, scores: list[AgentScore],
                    blind: bool = False) -> dict[str, float]:
        mapping: dict[str, str] | None = None
        if blind:
            scores, mapping = anonymize_scores(scores)
        raise NotImplementedError(
            "Jev wiring is for the live loop (key in user .env). "
            "Backtests use fund.decision.fusion.fuse_desks. "
            "When wired: send the (possibly blinded) scores to Jev, then "
            "deanonymize_convictions() the result with the mapping."
        )
        # When implemented:
        #   conv = <jev Score/Choice call on scores>
        #   return deanonymize_convictions(conv, mapping) if mapping else conv
