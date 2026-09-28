"""v2 research swarm: specialized agents, each a deterministic,
backtest-safe function of point-in-time data.

Each agent embodies one economic rationale and emits AgentScore rows
(ticker, agent, z-score, confidence, rationale, as_of). The judgment layer
(decision/fusion.py, Jev-shaped) fuses them into convictions. The future
LLM swarm plugs into the same contract: same inputs (data <= as_of),
same outputs.

Walk-forward discipline: every agent slices `closes.loc[:as_of]` FIRST.
Nothing past T is visible. Agents take an optional `sectors` map for
sector-relative features.

Agents:
  trend      -- multi-horizon momentum + trend alignment + 52w-high
                proximity + sector momentum (the "technicals" desk)
  reversal   -- short-term mean reversion (the contrarian desk;
                negatively correlated with trend by design)
  qualvol    -- low / contracting / downside volatility (the risk desk)
  flow       -- volume participation and accumulation (the flow desk)
Regime (macro desk) is separate: market_regime() in decision/fusion.py
returns a portfolio-level risk factor, not per-ticker scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fund.config import FundConfig
from fund.research.schema import AgentScore

SQRT252 = np.sqrt(252)


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10 or s.std() == 0:
        return pd.Series(dtype=float)
    return (s - s.mean()) / s.std()


def _hist(closes: pd.DataFrame, as_of, min_days: int) -> pd.DataFrame | None:
    h = closes.loc[:as_of]
    return h if len(h) >= min_days else None


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


# ---------------------------------------------------------------- trend
def trend_scores(
    closes: pd.DataFrame, as_of, cfg: FundConfig, sectors: dict | None = None
) -> dict[str, float]:
    hist = _hist(closes, as_of, 260)
    if hist is None:
        return {}
    px = hist.iloc[-1]
    f = {}
    f["mom6"] = _zscore(hist.iloc[-22] / hist.iloc[-148] - 1)          # 126d skip 21d
    f["mom3"] = _zscore(hist.iloc[-6] / hist.iloc[-69] - 1)            # 63d skip 5d
    ma50, ma200 = hist.tail(50).mean(), hist.tail(200).mean()
    align = (px > ma50).astype(int) + (ma50 > ma200).astype(int)       # 0..2
    f["align"] = _zscore(align.astype(float))
    f["hi52"] = _zscore(px / hist.tail(252).max())                     # 52w-high proximity
    if sectors:
        sec = pd.Series(sectors)
        sec_ret = {}
        for s in sec.unique():
            members = [t for t in sec[sec == s].index if t in hist.columns]
            if len(members) >= 3:
                sec_ret[s] = (hist[members].iloc[-1] / hist[members].iloc[-64] - 1).median()
        sec_z = _zscore(pd.Series(sec_ret))
        f["secmom"] = pd.Series({t: sec_z.get(sectors.get(t), np.nan) for t in hist.columns})
    frame = pd.DataFrame(f).dropna(how="all")
    return frame.mean(axis=1, skipna=True).to_dict()


# ------------------------------------------------------------- reversal
def reversal_scores(closes: pd.DataFrame, as_of, cfg: FundConfig) -> dict[str, float]:
    hist = _hist(closes, as_of, 80)
    if hist is None:
        return {}
    px = hist.iloc[-1]
    ma50 = hist.tail(50).mean()
    f = {}
    f["dist50"] = _zscore(-(px / ma50 - 1))        # below MA50 scores +
    f["rev5"] = _zscore(-(hist.iloc[-1] / hist.iloc[-6] - 1))  # down week scores +
    frame = pd.DataFrame(f).dropna(how="all")
    return frame.mean(axis=1, skipna=True).to_dict()


# ------------------------------------------------------------- qualvol
def qualvol_scores(closes: pd.DataFrame, as_of, cfg: FundConfig) -> dict[str, float]:
    hist = _hist(closes, as_of, 90)
    if hist is None:
        return {}
    rets = hist.pct_change()
    v63 = rets.tail(63).std() * SQRT252
    v21 = rets.tail(21).std() * SQRT252
    downside = rets.tail(63)[rets.tail(63) < 0].std() * SQRT252
    f = {}
    f["lowvol"] = _zscore(-v63)                    # calm names score +
    f["volcontract"] = _zscore(-(v21 / v63 - 1))   # contracting vol scores +
    f["nodown"] = _zscore(-downside)               # no downside jumps scores +
    frame = pd.DataFrame(f).dropna(how="all")
    return frame.mean(axis=1, skipna=True).to_dict()


# ----------------------------------------------------------------- flow
def flow_scores(
    closes: pd.DataFrame, volumes: pd.DataFrame, as_of, cfg: FundConfig
) -> dict[str, float]:
    hist_c = _hist(closes, as_of, 260)
    hist_v = volumes.loc[:as_of]
    if hist_c is None or len(hist_v) < 260:
        return {}
    f = {}
    v21 = hist_v.tail(21).mean()
    v63 = hist_v.tail(63).mean()
    f["voltrend"] = _zscore(v21 / v63 - 1)                       # rising participation
    rets = hist_c.pct_change().tail(63)
    up = hist_v.tail(63)[rets > 0].sum()
    dn = hist_v.tail(63)[rets < 0].sum().replace(0, np.nan)
    f["accum"] = _zscore(up / dn)                                # accumulation
    dv = hist_c * hist_v
    f["dvgrow"] = _zscore(dv.tail(63).median() / dv.tail(252).median() - 1)
    frame = pd.DataFrame(f).dropna(how="all")
    return frame.mean(axis=1, skipna=True).to_dict()


AGENTS = {
    "trend": trend_scores,
    "reversal": reversal_scores,
    "qualvol": qualvol_scores,
    "flow": flow_scores,
}


def score_universe(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    as_of,
    cfg: FundConfig,
    sectors: dict | None = None,
) -> list[AgentScore]:
    """Run the full swarm. One AgentScore per (ticker, agent)."""
    liquid = liquid_tickers(closes, volumes, as_of, cfg)
    as_of_date = pd.to_datetime(as_of).date()
    out: list[AgentScore] = []
    for name, fn in AGENTS.items():
        try:
            if name == "trend":
                scores = fn(closes, as_of, cfg, sectors)
            elif name == "flow":
                scores = fn(closes, volumes, as_of, cfg)
            else:
                scores = fn(closes, as_of, cfg)
        except Exception as e:  # noqa: BLE001
            # Loud, not silent: a dead agent is a missing signal, and the
            # fusion must never silently run on a subset of the swarm.
            print(f"  AGENT FAILURE {name} @ {as_of_date}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        for t, z in scores.items():
            if t in liquid:
                out.append(
                    AgentScore(
                        ticker=t,
                        agent=name,
                        score=float(z),
                        confidence=0.5,
                        rationale=f"{name}_z={z:+.2f}",
                        as_of=as_of_date,
                    )
                )
    return out


# ------------------------------------------------- v1 (comparison only)
def score_universe_v1(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    as_of,
    cfg: FundConfig,
    sectors: dict | None = None,
) -> list[AgentScore]:
    """Original v1 baseline: 0.7*momentum + 0.3*lowvol. Kept to measure
    whether the swarm actually adds anything."""
    liquid = liquid_tickers(closes, volumes, as_of, cfg)
    hist = closes.loc[:as_of]
    as_of_date = pd.to_datetime(as_of).date()
    if len(hist) < cfg.momentum_lookback + cfg.momentum_skip + 5:
        return []
    mom = _zscore(hist.iloc[-1 - cfg.momentum_skip] / hist.iloc[-1 - cfg.momentum_skip - cfg.momentum_lookback] - 1)
    lv = _zscore(-(hist.pct_change().tail(cfg.vol_window).std() * SQRT252))
    common = liquid & set(mom.dropna().index) & set(lv.dropna().index)
    return [
        AgentScore(t, "v1-blend", float(cfg.w_momentum * mom[t] + cfg.w_lowvol * lv[t]),
                   0.5, f"mom={mom[t]:+.2f} lv={lv[t]:+.2f}", as_of_date)
        for t in sorted(common)
    ]
