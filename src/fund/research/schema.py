"""Fixed output schema for research agents.

Every agent -- heuristic v1 or LLM swarm later -- emits AgentScore rows.
The decision layer only ever sees these, never raw prices, which keeps the
walk-forward discipline in one place: scores are computed from data with
`as_of <= T`, full stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class AgentScore:
    ticker: str
    agent: str          # e.g. "momentum", "lowvol", "llm-technicals"
    score: float        # higher = more attractive; cross-sectionally z-scored
    confidence: float   # 0..1, agent's self-assessed confidence
    rationale: str      # short human-readable reason (auditable)
    as_of: date
