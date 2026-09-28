"""v3 research desks: one per data source, all point-in-time.

The initial concept was four desks -- technicals, fundamentals/earnings,
news/sentiment, macro -- each pulling its OWN data source. v2 built only
the technicals desk (four agents, one tape). v3.1 runs the full swarm on
100% free sources:

  technicals   yfinance prices (unchanged)
  fundamentals SEC EDGAR XBRL companyfacts (10-K facts, filing-date gated)
  filings      EDGAR 8-Ks: Item 2.02 earnings drift (PEAD, the ai-hedge-fund
               steal) + abnormal 8-K attention x price confirmation. The
               market's 2-day reaction to the 8-K is the surprise proxy --
               no estimates feed exists for free, so the tape is the judge.
  macro        FRED (free key): yield-curve slope, CPI trend, unemployment
               trend -> portfolio-level regime factor, not stock picking.

Walk-forward discipline (the non-negotiable): every desk filters on
filing/event dates FIRST -- `filing_date <= as_of`, full stop. A reaction
is only scored once it is observable inside the as-of window.

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
    "fundamentals": "fundamentals",
    "filings": "filings",
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


# ---------------------------------------------------------------- filings
def filings_scores(fd, tickers: list[str], as_of, closes: pd.DataFrame,
                   params: dict) -> dict[str, tuple[float, float, str]]:
    """8-K filings desk: earnings drift + filing attention (free data).

    Two sub-signals, one desk:
    (a) PEAD (the ai-hedge-fund steal): a FRESH Item 2.02 8-K (filed within
        pead_window_days of as_of) drifts in the direction of the market's
        own 2-day reaction to the filing. No estimates feed exists for
        free, so the tape is the surprise proxy: |2-day CAR| < 0.5% is no
        verdict and emits nothing. The reaction must be observable inside
        the as-of window -- an 8-K filed today has no readable reaction yet.
    (b) Attention: abnormal 8-K filing rate (63d count vs 252d baseline)
        times the sign of the 63d price return -- heavy filings + rising
        price is positive attention, the reverse negative. Deterministic,
        no NLP.
    Desk score = pead_weight * z(pead) + attn_weight * z(attention),
    z-scored cross-sectionally. A ticker with neither sub-signal has no row.
    """
    as_of_d = _as_date(as_of)
    window = int(params.get("pead_window_days", 21))
    cap = float(params.get("pead_cap", 2.0))
    w_pead = float(params.get("pead_weight", 0.6))
    w_attn = float(params.get("attn_weight", 0.4))
    attn_look = int(params.get("attn_lookback_days", 63))
    attn_base = int(params.get("attn_baseline_days", 252))
    hist = closes.loc[:as_of]

    pead_raw: dict[str, float] = {}
    pead_meta: dict[str, str] = {}
    attn_raw: dict[str, float] = {}
    attn_meta: dict[str, str] = {}

    for t in tickers:
        try:
            raw = fd.filing_events(t, limit=500)
        except Exception:
            continue
        # Parse each event date ONCE: pd.to_datetime per event was 87% of
        # a full-universe review (39k calls for 50 tickers). Tuples of
        # (date, kind) keep the hot loop on cheap isinstance checks.
        events: list[tuple[date, str]] = []
        for e in raw:
            d = _as_date(e.get("date"))
            if d is not None and d <= as_of_d:
                events.append((d, e.get("kind")))
        if not events or t not in hist.columns:
            continue
        px = hist[t].dropna()
        if len(px) < attn_look + 2:
            continue

        # (a) PEAD on fresh earnings 8-Ks
        earn_dates = [d for d, k in events if k == "earnings"]
        if earn_dates:
            filed = max(earn_dates)
            age = (as_of_d - filed).days
            if age <= window:
                try:
                    pos = px.index.searchsorted(pd.to_datetime(filed))
                    pos = min(pos, len(px) - 1)
                    if pos + 1 < len(px):  # reaction observable in-window
                        car = float(px.iloc[pos + 1] / px.iloc[pos] - 1)
                        if abs(car) >= 0.005:
                            pead_raw[t] = float(np.sign(car)
                                                * min(abs(car) / 0.04, cap))
                            pead_meta[t] = (
                                f"PEAD: 8-K(2.02) filed {filed} ({age}d ago), "
                                f"2d CAR {car:+.1%}")
                except (KeyError, IndexError, TypeError):
                    pass

        # (b) abnormal 8-K attention x price direction
        dates = sorted(d for d, _ in events)
        n_recent = sum(1 for d in dates if (as_of_d - d).days <= attn_look)
        n_base = sum(1 for d in dates if (as_of_d - d).days <= attn_base)
        if n_base >= 2:
            expected = n_base * attn_look / attn_base
            abn = (n_recent - expected) / max(expected ** 0.5, 1.0)
            direction = float(np.sign(
                px.iloc[-1] / px.iloc[-min(attn_look + 1, len(px))] - 1))
            attn_raw[t] = abn * direction
            attn_meta[t] = (
                f"filings: abnormal 8-K attention "
                f"({'up' if direction > 0 else 'down'} {attn_look}d price)")

    z_pead = _z(pd.Series(pead_raw))
    z_attn = _z(pd.Series(attn_raw))
    # a zero-weighted leg must not create rows on its own
    active = set()
    if w_pead > 0:
        active |= set(z_pead.index)
    if w_attn > 0:
        active |= set(z_attn.index)
    combined = {t: w_pead * float(z_pead.get(t, 0.0))
                + w_attn * float(z_attn.get(t, 0.0))
                for t in active}
    z = _z(pd.Series(combined))
    out: dict[str, tuple[float, float, str]] = {}
    for t in z.index:
        bits = []
        if t in pead_meta:
            bits.append(pead_meta[t])
        if t in attn_meta:
            bits.append(attn_meta[t])
        out[t] = (float(z[t]), 0.6, "; ".join(bits))
    return out


# ---------------------------------------------------------------- macro
def _dstr(x) -> str:
    """ISO date string for chronological comparison; '' when missing.

    Lets the macro desk filter histories with plain string comparison
    instead of pd.to_datetime per observation. Missing dates sort before
    everything, so they never leak into the newest-first indexing.
    """
    s = str(x)[:10] if x is not None else ""
    return s if len(s) == 10 else ""


def macro_factor(fd, as_of, params: dict) -> float:
    """Portfolio-level macro regime factor in [0.5, 1.0].

    Blends three slow-moving macro reads -- yield-curve slope, inflation
    trend, labor trend -- into a risk-on/risk-off factor. This does NOT pick
    stocks; it scales the whole book, and blends with (not replaces) the
    price-based market_regime() via blend_with_price_regime.
    """
    as_of_d = _as_date(as_of)
    as_of_s = as_of_d.isoformat() if as_of_d else ""
    reads: list[float] = []  # each in [0, 1], 1 = risk-on

    try:
        # ISO "YYYY-MM-DD" strings compare chronologically -- no per-row
        # pd.to_datetime needed (~500 calls per review saved).
        curves = [c for c in fd.yield_curve_history(limit=400)
                  if _dstr(c.get("date")) <= as_of_s]
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
               if _dstr(c.get("date")) <= as_of_s]
        if len(cpi) >= 12:
            yoy = lambda c: c.get("yoy_change") or c.get("yoy") or 0
            # disinflation = risk-on
            reads.append(float(np.clip(0.5 - (yoy(cpi[0]) - yoy(cpi[11])) / 4.0,
                                       0, 1)))
    except Exception:
        pass

    try:
        unrate = [u for u in fd.unemployment_history(limit=48)
                  if _dstr(u.get("date")) <= as_of_s]
        if len(unrate) >= 4:
            # rising unemployment = risk-off (1pp rise over 3m -> full off)
            reads.append(float(np.clip(
                0.5 - (unrate[0]["rate"] - unrate[3]["rate"]) / 1.0, 0, 1)))
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
    existing deterministic agents; fundamentals/filings from the free SEC
    stack; macro from FRED. A desk that errors is LOUD (same rule as
    agents.py) -- fusion must never silently run short a desk.
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
        try:
            for t, (z, conf, why) in fundamentals_scores(
                    fd, liquid, as_of, params.get("fundamentals", {})).items():
                out.append(AgentScore(t, "fundamentals", z, conf, why, as_of_d))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE fundamentals @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    if "filings" in desk_names:
        try:
            for t, (z, conf, why) in filings_scores(
                    fd, liquid, as_of, closes,
                    params.get("filings", {})).items():
                out.append(AgentScore(t, "filings", z, conf, why, as_of_d))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE filings @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    macro_f = 1.0
    if "macro" in desk_names:
        try:
            macro_f = macro_factor(fd, as_of, params.get("macro", {}))
        except Exception as e:  # noqa: BLE001
            print(f"  DESK FAILURE macro @ {as_of_d}: "
                  f"{type(e).__name__}: {e}", flush=True)

    return out, macro_f
