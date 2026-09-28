"""Walk-forward weekly simulator with frictions.

Methodology (the anti-Baskets checklist):
  * Every decision at review date T uses only data with date <= T.
    Scores are computed on `closes.loc[:T]`; the constructor never sees
    the future.
  * No in-sample optimization anywhere: entry/exit rules and weights are
    fixed policy in FundConfig, not fit on the backtest window.
  * Frictions: slippage (bps per side, against the open) on every fill;
    commission configurable (0 for Alpaca).
  * Rebalance is real: weekly review -> orders -> executed at the NEXT
    trading day's open, exactly like the live loop will do on Mondays.
  * Prices are total-return adjusted at ingest, so dividends are included.

Simplifications (documented, not hidden):
  * Stop-loss is evaluated on daily closes; exits fill at the next open.
  * Position sizes are computed from Friday's close but filled at Monday's
    open -- the slippage model absorbs the gap.
  * No partial-fill / liquidity-impact modeling beyond the dollar-volume
    filter in research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from fund.config import FundConfig
from fund.decision.fusion import fuse_scores, fuse_scores_regime, market_regime
from fund.decision.portfolio import PositionState, construct_targets


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    review_count: int


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    daily = equity.pct_change().dropna()
    n = len(equity)
    years = n / 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    # round trips
    wins, losses, holds = 0, 0, []
    if not trades.empty:
        exits = trades[trades["side"] == "SELL"].groupby("ticker")
        for t, g in exits:
            buys = trades[(trades["ticker"] == t) & (trades["side"] == "BUY")]
            if buys.empty:
                continue
            # crude round-trip: total buy cost vs total sell proceeds
            pnl = g["value"].sum() - buys["value"].sum()
            # (ignores partial overlaps; fine for a v1 stat)
            if pnl > 0:
                wins += 1
            else:
                losses += 1
    win_rate = wins / (wins + losses) if (wins + losses) else 0.0
    turnover = float(trades["value"].abs().sum()) if not trades.empty else 0.0
    return {
        "total_return": round(float(total_ret), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd), 4),
        "trades": int(len(trades)),
        "round_trips": wins + losses,
        "win_rate": round(float(win_rate), 3),
        "turnover_x": round(turnover / equity.iloc[0], 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def run_backtest(
    panel: dict[str, pd.DataFrame],
    sectors: dict[str, str],
    score_fn,
    cfg: FundConfig,
    start: str,
    end: str,
) -> BacktestResult:
    closes_full = panel["close"]
    opens_full = panel["open"]
    volumes_full = panel["volume"]

    start_d, end_d = pd.to_datetime(start), pd.to_datetime(end)
    mask = (closes_full.index >= start_d) & (closes_full.index <= end_d)
    closes, opens = closes_full.loc[mask], opens_full.loc[mask]
    volumes = volumes_full.loc[mask]
    dates = closes.index

    # warmup counts trading days available in the FULL panel, not the window
    hist_offset = closes_full.index.get_loc(dates[0])

    # trading calendar: review = last trading day of each ISO week
    weeks: dict[tuple, list] = {}
    for d in dates:
        iso = d.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(d)
    ordered = [weeks[k] for k in sorted(weeks)]
    review_days = {w[-1] for w in ordered}
    trade_day_after: dict = {}
    all_dates = list(dates)
    pos_of = {d: i for i, d in enumerate(all_dates)}
    for w in ordered:
        nxt = pos_of[w[-1]] + 1
        if nxt < len(all_dates):
            trade_day_after[w[-1]] = all_dates[nxt]

    # MTM panel: forward-fill stale closes so a missing print != zero value
    mtm = closes.ffill()

    cash = cfg.initial_capital
    positions: dict[str, PositionState] = {}
    pending: dict[str, float] = {}  # ticker -> target shares, executed next open
    equity = pd.Series(index=dates, dtype=float)
    trade_log: list[dict] = []
    reviews = 0
    slip = cfg.slippage_bps / 10_000

    def portfolio_value(price_row) -> float:
        val = cash
        for t, p in positions.items():
            px = price_row.get(t)
            if px is not None and not pd.isna(px):
                val += p.shares * px
        return val

    for i, d in enumerate(all_dates):
        # ---- 1. execute pending orders at today's open ----
        if pending:
            open_row = opens.loc[d]
            # sells first to free cash
            items = sorted(pending.items(), key=lambda kv: kv[1] < 0, reverse=True)
            for t, target_shares in items:
                cur = positions[t].shares if t in positions else 0.0
                delta = target_shares - cur
                if abs(delta) < 1e-9:
                    continue
                px = open_row.get(t)
                if px is None or pd.isna(px) or px <= 0:
                    continue  # no print today; order dies (logged implicitly)
                side = "SELL" if delta < 0 else "BUY"
                fill = px * (1 - slip) if delta < 0 else px * (1 + slip)
                if delta > 0:
                    cost = delta * fill + cfg.commission_per_trade
                    if cost > cash:  # scale to available cash
                        delta = (cash - cfg.commission_per_trade) / fill
                        if delta <= 0:
                            continue
                        cost = delta * fill + cfg.commission_per_trade
                    cash -= cost
                else:
                    delta = max(delta, -cur)  # can't sell more than held
                    proceeds = -delta * fill - cfg.commission_per_trade
                    cash += proceeds
                    fill_shares = -delta
                    trade_log.append(
                        {"date": d.date(), "ticker": t, "side": side,
                         "shares": round(fill_shares, 4), "price": round(fill, 2),
                         "value": round(fill_shares * fill, 2)}
                    )
                    if t in positions:
                        if abs(positions[t].shares + delta) < 1e-9:
                            del positions[t]
                        else:
                            positions[t].shares += delta
                    continue
                # BUY bookkeeping (blended entry)
                if t in positions:
                    p = positions[t]
                    p.entry_price = (
                        p.shares * p.entry_price + delta * fill
                    ) / (p.shares + delta)
                    p.shares += delta
                else:
                    positions[t] = PositionState(t, delta, fill, d.date())
                trade_log.append(
                    {"date": d.date(), "ticker": t, "side": side,
                     "shares": round(delta, 4), "price": round(fill, 2),
                     "value": round(delta * fill, 2)}
                )
            pending = {}

        # ---- 2. mark to market ----
        equity.loc[d] = portfolio_value(mtm.loc[d].to_dict())

        # ---- 3. stop-loss scan on the close -> exit at next open ----
        if i + 1 < len(all_dates):
            close_row = closes.loc[d]
            for t, p in list(positions.items()):
                px = close_row.get(t)
                if px is None or pd.isna(px):
                    continue
                if px <= p.entry_price * (1 - cfg.stop_loss):
                    pending[t] = 0.0

        # ---- 4. weekly review: swarm -> Jev fusion -> targets -> orders ----
        if d in review_days and d in trade_day_after and hist_offset + i >= cfg.warmup_days:
            reviews += 1
            scored = score_fn(closes_full, volumes_full, d, cfg, sectors)
            regime = market_regime(closes_full, d, cfg)
            convictions = fuse_scores_regime(scored, cfg, regime.factor)
            if reviews % 25 == 1:
                print(f"  {d.date()} regime={regime.label} factor={regime.factor} ({regime.rationale})")
            price_today = closes.loc[d].dropna().to_dict()
            # positions view excludes names already pending a stop exit
            view = {t: p for t, p in positions.items() if t not in pending}
            targets = construct_targets(convictions, view, price_today, sectors, cfg,
                                        as_of=d.date(), blocked=set(pending),
                                        regime_factor=regime.factor)
            eq = equity.loc[d]
            new_pending: dict[str, float] = {}
            for t, w in targets.items():
                px = price_today.get(t)
                if px is None or px <= 0:
                    continue
                new_pending[t] = (w * eq) / px
            for t in view:
                if t not in new_pending:
                    new_pending[t] = 0.0
            # stop exits already in pending win over review targets
            for t in pending:
                new_pending[t] = 0.0
            # no-trade band: skip dust rebalances (full exits always execute)
            pending = {}
            for t, s in new_pending.items():
                cur = positions[t].shares if t in positions else 0.0
                if abs(s - cur) < 1e-9:
                    continue
                px = price_today.get(t, 0) or 0
                if s > 0 and t in positions and abs(s - cur) * px < cfg.min_trade_value:
                    continue
                pending[t] = s

    equity = equity.dropna()
    trades = pd.DataFrame(trade_log)
    return BacktestResult(
        equity=equity,
        trades=trades,
        metrics=compute_metrics(equity, trades),
        review_count=reviews,
    )
