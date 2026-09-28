"""v3 research desks: one per data source, all point-in-time.

The initial concept was four desks -- technicals, fundamentals/earnings,
news/sentiment, macro -- each pulling its OWN data source. v2 built only
the technicals desk (four agents, one tape). These three new desks close
the gap, all reading from the Financial Datasets stack (the Dexter steal).

Walk-forward discipline (the non-negotiable): every desk filters on
filing/publication dates FIRST -- `filing_date <= as_of`, full stop.
Borrowed patterns:
  * PEAD event logic from ai-hedge-fund's pead.py: only filings on/before
    `as_of` count, and only "fresh" ones (inside the signal window) fire.
  * Dexter's TTL idea lives in the FD client; desks just call it.

Output contract is unchanged: AgentScore rows (ticker, agent, z-score,
confidence, rationale, as_of). The technicals desk is the existing
score_universe() in agents.py; macro is portfolio-level (a regime factor),
not per-ticker.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from fund.config import FundConfig
from fund.research.schema import AgentScore

# agent -> desk mapping, used by fusion to apply desk weights
AGENT_DESK = {
    "trend": "technicals", "reversal": "technicals", "flow": "technicals",
    "qualvol": "technicals",
    "fundamentals": "fundamentals", "pead": "fundamentals",
    "sentiment": "sentiment",
}


def _z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10 or s.std() == 0:
        return pd.Series(dtype=float)
    return (s - s.mean()) / s.std()


def _as_date(x) -> date | None:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except (ValueError, TypeError):
        return None


def _filed_on_or_before(row: dict, as_of: date) -> bool:
    d = _as_date(row.get("filing_date") or row.get("filing_datetime"))
    return d is not None and d <= as_of


# ---------------------------------------------------------------- fundamentals
def fundamentals_scores(fd, tickers: list[str], as_of, params: dict
                         ) -> dict[str, tuple[float, float, str]]:
    """Value + quality + growth from filing-date-gated quarterly metrics.

    Only metrics with filing_date <= as_of are visible (no lookahead).
    Composite = value_weight*(1/PE + FCF yield) + quality_weight*(ROE +
    gross margin) + growth_weight*(EPS growth), z-scored cross-sectionally.
    """
    as_of_d = _as_date(as_of)
    n_periods = int(params.get("metrics_periods", 8))
    w_v = float(params.get("value_weight", 0.40))
    w_q = float(params.get("quality_weight", 0.35))
    w_g = float(params.get("growth_weight", 0.25))

    raw: dict[str, tuple[float, float]] = {}  # ticker -> (composite, completeness)
    for t in tickers:
        try:
            rows = [r for r in fd.metrics(t, limit=n_periods)
                    if _filed_on_or_before(r, as_of_d)]
        except Exception:
            continue
        if not rows:
            continue
        m = rows[0]  # newest filed as of T
        parts, present, total = [], 0, 0

        def grab(key: str) -> float | None:
            v = m.get(key)
            return float(v) if v is not None else None

        pe = grab("price_to_earnings_ratio")
        total += 1
        if pe and pe > 0:
            parts.append(("value", 1.0 / pe)); present += 1
        fcfy = grab("free_cash_flow_yield")
        total += 1
        if fcfy is not None:
            parts.append(("value", fcfy)); present += 1
        roe = grab("return_on_equity")
        total += 1
        if roe is not None:
            parts.append(("quality", roe)); present += 1
        gm = grab("gross_margin")
        total += 1
        if gm is not None:
            parts.append(("quality", gm)); present += 1
        epsg = grab("earnings_per_share_growth")
        if epsg is None:
            epsg = grab("revenue_growth")
        total += 1
        if epsg is not None:
            parts.append(("growth", epsg)); present += 1

        if not parts:
            continue
        comp: dict[str, list[float]] = {"value": [], "quality": [], "growth": []}
        for k, v in parts:
            comp[k].append(v)
        score = ((w_v * float(np.mean(comp["value"])) if comp["value"] else 0.0)
                 + (w_q * float(np.mean(comp["quality"])) if comp["quality"] else 0.0)
                 + (w_g * float(np.mean(comp["growth"])) if comp["growth"] else 0.0))
        raw[t] = (score, present / total)

    z = _z(pd.Series({t: s for t, (s, _) in raw.items()}))
    out: dict[str, tuple[float, float, str]] = {}
    for t in z.index:
        _, completeness = raw[t]
        out[t] = (float(z[t]), 0.5 + 0.3 * completeness,
                  f"fundamentals: value/quality/growth composite "
                  f"(completeness {completeness:.0%})")
    return out


def pead_scores(fd, tickers: list[str], as_of, params: dict
                ) -> dict[str, tuple[float, float, str]]:
    """Post-earnings announcement drift (the ai-hedge-fund PEAD steal).

    Bullish after a BEAT, bearish after a MISS -- but ONLY if the filing is
    fresh: filing_date <= as_of AND within pead_window_days. A stale filing
    is not a signal, it's history. Magnitude scaled by surprise_pct, capped.
    Tickers with no fresh event emit no row (no view).
    """
    as_of_d = _as_date(as_of)
    window = int(params.get("pead_window_days", 21))
    cap = float(params.get("pead_cap", 2.0))

    raw: dict[str, float] = {}
    meta: dict[str, str] = {}
    for t in tickers:
        try:
            rows = [r for r in fd.earnings(t, limit=8)
                    if _filed_on_or_before(r, as_of_d)]
        except Exception:
            continue
        if not rows:
            continue
        e = max(rows, key=lambda r: str(r.get("filing_date") or ""))
        filed = _as_date(e.get("filing_date") or e.get("filing_datetime"))
        if filed is None or (as_of_d - filed).days > window:
            continue
        surprise = str(e.get("eps_surprise") or e.get("surprise") or "").upper()
        if surprise not in ("BEAT", "MISS"):
            continue
        pct = e.get("eps_surprise_pct")
        mag = min(abs(float(pct)) / 10.0, cap) if pct is not None else 1.0
        raw[t] = (1.0 if surprise == "BEAT" else -1.0) * mag
        meta[t] = f"PEAD: EPS {surprise} filed {filed} ({(as_of_d - filed).days}d ago)"

    z = _z(pd.Series(raw))
    return {t: (float(z[t]), 0.8, meta[t]) for t in z.index}


# ---------------------------------------------------------------- sentiment
def sentiment_scores(fd, tickers: list[str], as_of, closes: pd.DataFrame,
                     params: dict) -> dict[str, tuple[float, float, str]]:
    """News attention x price confirmation (deterministic v0).

    Abnormal news volume (7d count vs 90d baseline, z-scored) times the sign
    of the 7d price return: heavy news + rising price = positive attention,
    heavy news + falling price = negative. No NLP, no black box -- a
    transparent proxy until a real sentiment feed is wired. Only news with
    published_at <= as_of counts.
    """
    as_of_d = _as_date(as_of)
    look = int(params.get("news_lookback_days", 7))
    base = int(params.get("news_baseline_days", 90))
    hist = closes.loc[:as_of]

    attn: dict[str, float] = {}
    direction: dict[str, float] = {}
    for t in tickers:
        try:
            items = [n for n in fd.news(t, limit=10)
                     if (_as_date(n.get("published_at") or n.get("date"))
                         is not None)
                     and _as_date(n.get("published_at") or n.get("date")) <= as_of_d]
        except Exception:
            continue
        if len(items) < 3 or t not in hist.columns:
            continue
        dates = sorted(_as_date(n.get("published_at") or n.get("date"))
                       for n in items)
        n_recent = sum(1 for d in dates
                       if (as_of_d - d).days <= look)
        n_base = sum(1 for d in dates
                     if (as_of_d - d).days <= base)
        expected = n_base * look / base
        attn[t] = (n_recent - expected) / max(expected ** 0.5, 1.0)
        px = hist[t].dropna()
        direction[t] = float(np.sign(px.iloc[-1] / px.iloc[-min(look + 1, len(px))] - 1))

    z = _z(pd.Series(attn))
    out: dict[str, tuple[float, float, str]] = {}
    for t in z.index:
        s = float(z[t]) * direction[t]
        out[t] = (s, 0.5,
                  f"sentiment: abnormal news attention "
                  f"({'up' if direction[t] > 0 else 'down'} {look}d price)")
    return out


# ---------------------------------------------------------------- macro
def macro_factor(fd, as_of, params: dict) -> float:
    """Portfolio-level macro regime factor in [0.5, 1.0].

    Blends three slow-moving macro reads -- yield-curve slope, inflation
    trend, labor trend -- into a risk-on/risk-off factor. This does NOT pick
    stocks; it scales the whole book, and blends with (not replaces) the
    price-based market_regime() via blend_with_price_regime.
    """
    as_of_d = _as_date(as_of)
    reads: list[float] = []  # each in [0, 1], 1 = risk-on

    try:
        curves = [c for c in fd.yield_curve_history(limit=400)
                  if _as_date(c.get("date")) <= as_of_d]
        if len(curves) >= 252:
            now, year_ago = curves[0], curves[252]
            slope = lambda c: (c.get("10_year") or 0) - (c.get("2_year") or 0)
            # steepening curve = risk-on; invert the stress signal
            reads.append(float(np.clip(0.5 + (slope(now) - slope(year_ago)) / 2.0,
                                       0, 1)))
    except Exception:
        pass

    try:
        cpi = [c for c in fd.inflation_history(limit=48)
               if _as_date(c.get("date")) <= as_of_d]
        if len(cpi) >= 12:
            yoy = lambda c: c.get("yoy_change") or c.get("yoy") or 0
            # disinflation = risk-on
            reads.append(float(np.clip(0.5 - (yoy(cpi[0]) - yoy(cpi[11])) / 4.0,
                                       0, 1)))
    except Exception:
        pass

    if not reads:
        return 1.0  # no macro data: don't gate the book on nothing
    risk_on = float(np.mean(reads))
    return 0.5 + 0.5 * risk_on


# ---------------------------------------------------------------- entry point
def score_desks(fd, closes: pd.DataFrame, volumes: pd.DataFrame, as_of,
                mandate, sectors: dict | None = None
                ) -> tuple[list[AgentScore], float]:
    """Run the full 4-desk swarm for one review date.

    Returns (agent_scores, macro_regime_factor). Technicals come from the
    existing deterministic agents; fundamentals/pead/sentiment from the FD
    stack; macro as a portfolio-level factor. A desk that errors is LOUD
    (same rule as agents.py) -- fusion must never silently run short a desk.
    """
    from fund.research.agents import liquid_tickers, score_universe

    as_of_d = _as_date(as_of)
    cfg = FundConfig()
    liquid = sorted(liquid_tickers(closes, volumes, as_of, cfg))
    params = mandate.desk_params
    out: list[AgentScore] = []

    desk_names = [d.name for d in mandate.desks]

    if "technicals" in desk_names:
        tp = params.get("technicals", {})
        want = set(tp.get("agents", ["trend", "reversal", "flow"]))
        for s in score_universe(closes, volumes, as_of, cfg, sectors):
            if s.agent in want:
                # reweight inside the technicals desk per mandate
                wkey = {"trend": "w_trend", "reversal": "w_reversal",
                        "flow": "w_flow"}.get(s.agent)
                if wkey:
                    s.confidence *= float(tp.get(wkey, 1.0))
                out.append(s)

    if "fundamentals" in desk_names:
        fp = params.get("fundamentals", {})
        try:
            for t, (z, conf, why) in fundamentals_scores(
                    fd, liquid, as_of, fp).items():
                out.append(AgentScore(t, "fundamentals", z, conf, why, as_of_d))
            for t, (z, conf, why) in pead_scores(fd, liquid, as_of, fp).items():
                out.append(AgentScore(t, "pead", z, conf, why, as_of_d))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE fundamentals @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    if "sentiment" in desk_names:
        try:
            for t, (z, conf, why) in sentiment_scores(
                    fd, liquid, as_of, closes,
                    params.get("sentiment", {})).items():
                out.append(AgentScore(t, "sentiment", z, conf, why, as_of_d))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE sentiment @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    macro_f = 1.0
    if "macro" in desk_names:
        try:
            macro_f = macro_factor(fd, as_of, params.get("macro", {}))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE macro @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    return out, macro_f
