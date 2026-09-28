"""Paper-trading planning logic. Pure functions, no I/O, unit-testable.

Turns fused target weights + current broker positions into an explicit
order plan, with the same no-trade band the backtester uses and hard
sanity checks that run before anything touches the broker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from fund.decision.portfolio import PositionState

# Per-order ceiling as a fraction of equity: the 3% position cap plus a
# small tolerance for price drift between Friday's close (sizing) and
# Monday's open (fill). Anything above this is rejected, not clipped.
MAX_ORDER_PCT_OF_EQUITY = 0.032


@dataclass
class PlannedOrder:
    symbol: str
    side: str            # "buy" | "sell"
    notional: Optional[float]  # buys: dollars
    qty: Optional[float]       # sells: shares
    est_value: float            # |delta| in dollars, for sanity checks
    reason: str


def most_recent_friday(dates, today: Optional[date] = None) -> date:
    """Latest date in `dates` (a DatetimeIndex) that is a Friday <= today."""
    today = today or date.today()
    ts = [d for d in dates if d.date() <= today and d.weekday() == 4]
    if not ts:
        raise ValueError("no Friday close found in the price panel")
    return max(ts).date()


def build_order_plan(
    targets: dict[str, float],
    positions: dict[str, dict],
    equity: float,
    prices: dict[str, float],
    min_trade_value: float = 500.0,
) -> list[PlannedOrder]:
    """Diff target weights against current positions -> order plan.

    targets: ticker -> target weight (fraction of equity), long-only.
    positions: ticker -> {qty, market_value, current_price} (broker view).
    Buys are notional ($) orders; sells are share-quantity orders, capped
    at the held quantity so a stale price can never create a short.
    """
    orders: list[PlannedOrder] = []
    held = {t.upper(): p for t, p in positions.items()}

    for t, w in targets.items():
        if w < 0:
            raise ValueError(f"negative target weight for {t}: {w} (long-only)")
        if w == 0:
            continue
        px = prices.get(t) or 0
        if px <= 0:
            continue  # no price -> can't size; skip loudly in the summary
        target_notional = w * equity
        pos = held.get(t.upper(), {})
        cur_mv = float(pos.get("market_value") or 0)
        delta = target_notional - cur_mv
        if abs(delta) < min_trade_value:
            continue  # no-trade band: skip dust rebalances
        if delta > 0:
            orders.append(PlannedOrder(
                symbol=t, side="buy", notional=round(delta, 2), qty=None,
                est_value=round(delta, 2),
                reason=f"raise to {w:.1%} of equity"))
        else:
            held_qty = float(pos.get("qty") or 0)
            if held_qty <= 0:
                continue
            sell_qty = min(-delta / px, held_qty)  # never sell more than held
            full = abs(sell_qty - held_qty) < 1e-9
            orders.append(PlannedOrder(
                symbol=t, side="sell", notional=None,
                qty=round(sell_qty, 6), est_value=round(-delta, 2),
                reason="full exit" if full else f"trim to {w:.1%} of equity"))

    # exits: held but absent from targets -> liquidate
    for t, pos in held.items():
        if t in {k.upper() for k in targets}:
            continue
        held_qty = float(pos.get("qty") or 0)
        if held_qty <= 0:
            continue
        orders.append(PlannedOrder(
            symbol=t, side="sell", notional=None,
            qty=round(held_qty, 6),
            est_value=round(float(pos.get("market_value") or 0), 2),
            reason="no longer a target: full exit"))

    orders.sort(key=lambda o: (o.side != "sell", o.symbol))
    return orders


def check_order_sanity(orders: list[PlannedOrder], equity: float,
                       positions: Optional[dict[str, dict]] = None) -> None:
    """Hard guards. Raises ValueError on the first violation -- the run
    aborts before any order is submitted."""
    if equity <= 0:
        raise ValueError(f"refusing to trade against non-positive equity {equity}")
    held = {t.upper(): float(p.get("qty") or 0) for t, p in (positions or {}).items()}
    for o in orders:
        if o.side not in ("buy", "sell"):
            raise ValueError(f"unknown side {o.side!r} for {o.symbol}")
        if o.est_value > MAX_ORDER_PCT_OF_EQUITY * equity:
            raise ValueError(
                f"{o.symbol}: order ${o.est_value:,.0f} exceeds "
                f"{MAX_ORDER_PCT_OF_EQUITY:.1%} of equity "
                f"(${MAX_ORDER_PCT_OF_EQUITY * equity:,.0f}) -- rejected")
        if o.side == "sell":
            if (o.qty or 0) <= 0:
                raise ValueError(f"{o.symbol}: sell with non-positive qty")
            if (o.qty or 0) > held.get(o.symbol.upper(), 0) + 1e-9:
                raise ValueError(
                    f"{o.symbol}: sell qty {o.qty} exceeds held "
                    f"{held.get(o.symbol.upper(), 0)} -- would go short")
        else:
            if (o.notional or 0) <= 0:
                raise ValueError(f"{o.symbol}: buy with non-positive notional")


def apply_position_cap(targets: dict[str, float], cap: float) -> dict[str, float]:
    """Hard-clip every target weight to `cap`.

    construct_targets tilts by conviction, which can nudge a name a hair
    above max_position_weight. The backtester keeps that behavior (its
    results are committed); the live loop applies this hard clip after so
    the user's 3% risk cap is never breached. Clipping only ever moves
    money to cash, never between names.
    """
    return {t: min(w, cap) for t, w in targets.items()}


def positions_from_snapshot(snapshot: dict[str, dict],
                            as_of: date) -> dict[str, PositionState]:
    """Rebuild PositionState for construct_targets from a saved snapshot.

    Unknown entry dates default to `as_of` (no immediate time-stop on
    positions we didn't open ourselves).
    """
    out: dict[str, PositionState] = {}
    for t, p in snapshot.items():
        try:
            ed = date.fromisoformat(str(p.get("entry_date", "")))
        except ValueError:
            ed = as_of
        out[t] = PositionState(
            ticker=t,
            shares=float(p.get("qty") or 0),
            entry_price=float(p.get("avg_entry_price") or p.get("entry_price") or 0),
            entry_date=ed,
        )
    return out


def summarize_plan(orders: list[PlannedOrder], targets: dict[str, float],
                   equity: float) -> str:
    lines = [f"equity ${equity:,.0f} | targets {len(targets)} names | "
             f"orders {len(orders)}"]
    buys = [o for o in orders if o.side == "buy"]
    sells = [o for o in orders if o.side == "sell"]
    lines.append(f"  buys:  {len(buys):>3}  ${sum(o.est_value for o in buys):>10,.0f}")
    lines.append(f"  sells: {len(sells):>3}  ${sum(o.est_value for o in sells):>10,.0f}")
    for o in orders:
        amt = f"${o.notional:,.0f} notional" if o.side == "buy" else f"{o.qty} sh"
        lines.append(f"    {o.side.upper():4} {o.symbol:8} {amt:>16}  ({o.reason})")
    top = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)[:10]
    lines.append("  top targets: " + ", ".join(f"{t} {w:.1%}" for t, w in top))
    return "\n".join(lines)
