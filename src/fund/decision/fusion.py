"""The judgment layer: turns agent scores into calibrated convictions.

Contract (Jev-compatible):  list[AgentScore] -> {ticker: conviction in [0,1]}.
When the Jev API key is available, `JevClient.convictions()` in jev.py
replaces `fuse_scores()` below with zero changes to the simulator or the
portfolio constructor -- same inputs, same outputs, same walk-forward
discipline.

Local fusion design (the stand-in until Jev is wired):
  1. Per ticker, take the confidence-weighted mean of agent z-scores.
  2. Map z -> probability via the normal CDF: conviction = Phi(z).
     A ticker one sigma above the swarm mean gets ~0.84; the median
     ticker gets 0.50. This is calibrated by construction -- convictions
     are percentiles of the swarm's joint view, not raw scores.
  3. Rank by conviction for selection; conviction also tilts position
     weights in the portfolio constructor (Jev's "Choice").

market_regime() is the macro desk: a portfolio-level risk factor from the
tradable universe itself (no external data needed). It scales target
weights, leaving the remainder as cash -- an explicit risk-on/risk-off
overlay instead of v1's implicit luck.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fund.config import FundConfig
from fund.research.schema import AgentScore

AGENT_WEIGHTS = {"trend": "w_trend", "reversal": "w_reversal",
                 "qualvol": "w_qualvol", "flow": "w_flow"}


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fuse_scores(scores: list[AgentScore], cfg: FundConfig) -> dict[str, float]:
    """Fuse swarm outputs into per-ticker convictions in [0, 1].

    Fixed-weight version: the Jev-contract reference implementation
    (list[AgentScore] -> {ticker: conviction}). The live strategy uses
    fuse_scores_regime; this stays as the interface baseline."""
    by_ticker: dict[str, list[tuple[float, float]]] = {}
    for s in scores:
        w_attr = AGENT_WEIGHTS.get(s.agent)
        w = float(getattr(cfg, w_attr, 0.0)) if w_attr else 1.0
        by_ticker.setdefault(s.ticker, []).append((s.score, s.confidence * w))
    out: dict[str, float] = {}
    for t, parts in by_ticker.items():
        num = sum(z * c for z, c in parts)
        den = sum(c for _, c in parts)
        out[t] = _phi(num / den) if den > 0 else 0.5
    return out


# Regime-aware fusion weights (v2.1). Documented priors, not fitted:
#   * momentum is pro-cyclical and crashes in stress (Daniel-Moskowitz) ->
#     trend weight falls as the regime factor falls
#   * short-term reversal strengthens in high-vol regimes -> reversal
#     weight rises as the regime factor falls
#   * volume confirmation is regime-agnostic -> flow ~constant
#   * low-vol (qualvol): excluded (weight 0). It is top-minus-bottom
#     NEGATIVE in both regimes in this universe (risk-on t=-8.4,
#     risk-off t=-4.3, 2020-2026 S&P current constituents). The textbook
#     low-vol premium inverts here: survivorship-biased universe +
#     mega-cap momentum regime means "calm" = "missed the rockets".
#     A broader universe or the real Jev layer may reintroduce it.
W_ON  = {"trend": 0.55, "flow": 0.35, "reversal": 0.10, "qualvol": 0.0}
W_OFF = {"trend": 0.15, "flow": 0.30, "reversal": 0.55, "qualvol": 0.0}


def fuse_scores_regime(
    scores: list[AgentScore], cfg: FundConfig, regime_factor: float
) -> dict[str, float]:
    """Regime-aware fusion: interpolate agent weights between the risk-on
    and risk-off priors by the regime factor (0.5 -> W_OFF, 1.0 -> W_ON).
    Smooth in the factor: no bucket jumps, no extra parameters."""
    f = min(max((regime_factor - 0.5) / 0.5, 0.0), 1.0)
    w = {a: W_OFF[a] + (W_ON[a] - W_OFF[a]) * f
         for a in ("trend", "flow", "reversal", "qualvol")}
    by_ticker: dict[str, list[tuple[float, float]]] = {}
    for s in scores:
        wa = w.get(s.agent, 1.0)  # unknown agents (e.g. v1-blend): weight 1
        by_ticker.setdefault(s.ticker, []).append((s.score, s.confidence * wa))
    out: dict[str, float] = {}
    for t, parts in by_ticker.items():
        num = sum(z * c for z, c in parts)
        den = sum(c for _, c in parts)
        out[t] = _phi(num / den) if den > 0 else 0.5
    return out


@dataclass
class RegimeResult:
    factor: float        # multiplies all target weights; remainder is cash
    label: str           # risk-on / neutral / risk-off
    rationale: str


def market_regime(closes: pd.DataFrame, as_of, cfg: FundConfig) -> RegimeResult:
    """Risk overlay from the tradable universe itself, strictly as of T.

    factor: 1.0 in a healthy market, down to `regime_floor` when the
    trend is broken, breadth collapses, or volatility spikes.
    """
    hist = closes.loc[:as_of]
    need = max(cfg.regime_trend_ma, 252) + 5
    if len(hist) < need:
        return RegimeResult(1.0, "neutral", "insufficient history")
    rets = hist.pct_change()
    mkt_rets = rets.median(axis=1)                      # median-based market index
    idx = (1 + mkt_rets.fillna(0)).cumprod()
    trend_up = bool(idx.iloc[-1] > idx.tail(cfg.regime_trend_ma).mean())
    breadth = float((hist.iloc[-1] > hist.tail(cfg.regime_breadth_ma).mean()).mean())
    v21 = mkt_rets.tail(21).std() * np.sqrt(252)
    v_base = mkt_rets.rolling(21).std().tail(252).median() * np.sqrt(252)
    stress = bool(v21 > cfg.regime_vol_mult * v_base)

    factor, flags = 1.0, []
    if not trend_up:
        factor *= 0.7
        flags.append("trend-broken")
    if stress:
        factor *= 0.8
        flags.append("vol-spike")
    if breadth < 0.35:
        factor *= 0.85
        flags.append(f"breadth-weak({breadth:.0%})")
    factor = max(factor, cfg.regime_floor)
    label = "risk-on" if factor >= 0.99 else ("risk-off" if factor <= 0.6 else "neutral")
    rationale = "healthy" if not flags else "+".join(flags)
    return RegimeResult(round(factor, 3), label, rationale)


# ---------------------------------------------------------------- v3: desk fusion
def fuse_desks(
    scores: list[AgentScore],
    desk_weights: dict[str, float],
    regime_factor: float = 1.0,
) -> dict[str, float]:
    """Fuse the 4-desk swarm into convictions.

    Two-level fusion:
      1. Within the technicals desk, agent weights interpolate between the
         risk-on / risk-off priors by the regime factor (the validated v2.1
         logic: momentum crashes in stress, reversal wakes up).
      2. Across desks, the mandate's documented priors apply (fixed -- the
         regime moves the technicals mix and the portfolio-level factor,
         not the desk weights; fewer moving parts, fewer ways to fool
         ourselves).

    Desks absent from a ticker simply don't contribute (weights renormalize
    over desks present). Unknown agents fall back to weight 1.0.
    """
    from fund.research.desks import AGENT_DESK

    f = min(max((regime_factor - 0.5) / 0.5, 0.0), 1.0)
    tech_w = {a: W_OFF[a] + (W_ON[a] - W_OFF[a]) * f
              for a in ("trend", "flow", "reversal", "qualvol")}

    # ticker -> desk -> list[(z, confidence*agent_weight)]
    grid: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for s in scores:
        desk = AGENT_DESK.get(s.agent)
        if desk is None:
            continue  # not part of any desk: not fused
        wa = tech_w.get(s.agent, 1.0)
        grid.setdefault(s.ticker, {}).setdefault(desk, []).append(
            (s.score, s.confidence * wa))

    out: dict[str, float] = {}
    for t, desks in grid.items():
        desk_z: dict[str, float] = {}
        for d, parts in desks.items():
            num = sum(z * c for z, c in parts)
            den = sum(c for _, c in parts)
            if den > 0:
                desk_z[d] = num / den
        wsum = sum(desk_weights[d] for d in desk_z)
        if wsum <= 0:
            out[t] = 0.5
            continue
        z = sum(desk_z[d] * desk_weights[d] for d in desk_z) / wsum
        out[t] = _phi(z)
    return out


def blend_regime(price_factor: float, macro_factor: float,
                 blend: float = 0.5, floor: float = 0.5) -> float:
    """Combine the price-based regime factor with the macro desk's factor.

    blend=0 -> price only, 1 -> macro only. Floored like market_regime.
    """
    b = min(max(blend, 0.0), 1.0)
    return max((1 - b) * price_factor + b * macro_factor, floor)


# ---------------------------------------------------------------- backtest anonymization
def anonymize_scores(
    scores: list[AgentScore],
) -> tuple[list[AgentScore], dict[str, str]]:
    """Strip identity before the judge sees backtest data.

    Stolen from ai-hedge-fund: an LLM judge trained after the backtest
    window may *remember* how a named company did, and that memory scores
    as skill. So in backtests the Jev path must see tickers as T-0001...
    labels, no industries, no calendar dates (as_of -> None), and rationales
    scrubbed of the ticker string.

    Caveat (theirs too): distinctive numbers can still give a large company
    away. Anonymization reduces recall, it doesn't remove it -- a window
    after the judge's training cutoff is the cleanest read. Live runs are
    unaffected and name the company.
    """
    tickers = sorted({s.ticker for s in scores})
    mapping = {t: f"T-{i:04d}" for i, t in enumerate(tickers)}
    blind: list[AgentScore] = []
    for s in scores:
        rationale = s.rationale.replace(s.ticker, mapping[s.ticker])
        blind.append(AgentScore(
            ticker=mapping[s.ticker],
            agent=s.agent,
            score=s.score,
            confidence=s.confidence,
            rationale=rationale,
            as_of=None,  # type: ignore[arg-type]
        ))
    return blind, mapping


def deanonymize_convictions(blind_conv: dict[str, float],
                            mapping: dict[str, str]) -> dict[str, float]:
    """Map T-0001 labels back to tickers after the blind judge rules."""
    reverse = {v: k for k, v in mapping.items()}
    return {reverse[k]: v for k, v in blind_conv.items() if k in reverse}
