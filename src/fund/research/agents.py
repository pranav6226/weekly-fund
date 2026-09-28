"""v1 research agents: deterministic, backtest-safe price-based scorers.

These stand in for the LLM research swarm + Jev layer while we validate the
pipeline machinery (data -> scores -> portfolio -> fills -> accounting).
They are deliberately simple: momentum + low-volatility, the two most
replicated cross-sectional premia. The swarm replaces `score_universe()`
later; the portfolio constructor and simulator do not change.

Walk-forward discipline: every function takes `as_of` and slices
`closes.loc[:as_of]` FIRST. Nothing past T is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fund.config import FundConfig
from fund.research.schema import AgentScore


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10 or s.std() == 0:
        return pd.Series(dtype=float)
    return (s - s.mean()) / s.std()


def momentum_scores(closes: pd.DataFrame, as_of, cfg: FundConfig) -> dict[str, float]:
    """126d return skipping the most recent 21d, cross-sectional z-score."""
    hist = closes.loc[:as_of]
    if len(hist) < cfg.momentum_lookback + cfg.momentum_skip + 5:
        return {}
    p1 = hist.iloc[-1 - cfg.momentum_skip]
    p0 = hist.iloc[-1 - cfg.momentum_skip - cfg.momentum_lookback]
    ret = p1 / p0 - 1
    return _zscore(ret).to_dict()


def lowvol_scores(closes: pd.DataFrame, as_of, cfg: FundConfig) -> dict[str, float]:
    """Lower 63d realized vol scores higher (z-scored, sign flipped)."""
    hist = closes.loc[:as_of]
    if len(hist) < cfg.vol_window + 5:
        return {}
    rets = hist.pct_change().tail(cfg.vol_window)
    vol = rets.std() * np.sqrt(252)
    return (-_zscore(vol)).to_dict()


def liquid_tickers(
    closes: pd.DataFrame, volumes: pd.DataFrame, as_of, cfg: FundConfig
) -> set[str]:
    """Price and dollar-volume filters, evaluated strictly as of T."""
    hist_c = closes.loc[:as_of]
    hist_v = volumes.loc[:as_of]
    if len(hist_c) < 63:
        return set()
    price = hist_c.iloc[-1]
    dollar_vol = (hist_c * hist_v).tail(63).median()
    ok = (price >= cfg.min_price) & (dollar_vol >= cfg.min_dollar_volume)
    return set(ok[ok].index)


def score_universe(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    as_of,
    cfg: FundConfig,
) -> list[AgentScore]:
    """Run all v1 agents and combine into a single score per ticker."""
    liquid = liquid_tickers(closes, volumes, as_of, cfg)
    mom = momentum_scores(closes, as_of, cfg)
    lv = lowvol_scores(closes, as_of, cfg)
    common = liquid & set(mom) & set(lv)

    out: list[AgentScore] = []
    for t in sorted(common):
        score = cfg.w_momentum * mom[t] + cfg.w_lowvol * lv[t]
        out.append(
            AgentScore(
                ticker=t,
                agent="v1-blend",
                score=float(score),
                confidence=0.5,
                rationale=(
                    f"mom_z={mom[t]:+.2f} (126d skip 21d), "
                    f"lowvol_z={lv[t]:+.2f} (63d)"
                ),
                as_of=pd.to_datetime(as_of).date(),
            )
        )
    return out
