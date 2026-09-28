"""Portfolio construction. THE shared decision function.

Used identically by the backtester and the live loop, so a backtest result
describes the exact logic that will trade. Pure function: scores +
current positions + today's prices -> target weights. No I/O, no dates
beyond `as_of`, trivially unit-testable.

Rules (v2):
  * rank all scored tickers by CONVICTION (fused swarm output in [0,1]);
    candidates = top `entry_rank`
  * exits: rank fell past `exit_rank` (hysteresis), stop-loss hit, time-stop
    (held longer than `max_hold_days` -- the thesis must work), or ticker
    left the scored set (illiquid / missing data)
  * entries: fill up to `max_positions` from candidates, highest conviction first
  * weights: equal-weight base, tilted by conviction (0.75 + 0.5*conviction,
    renormalized so the tilt only redistributes), capped at
    `max_position_weight`; then `regime_factor` scales everything (remainder
    is cash); then enforce `max_sector_weight` by dropping the
    lowest-conviction name in any over-weight sector until it fits
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from fund.config import FundConfig


@dataclass
class PositionState:
    ticker: str
    shares: float
    entry_price: float
    entry_date: date


def construct_targets(
    convictions: dict[str, float],
    positions: dict[str, PositionState],
    prices_today: dict[str, float],
    sectors: dict[str, str],
    cfg: FundConfig,
    as_of: date,
    blocked: set[str] | None = None,
    regime_factor: float = 1.0,
) -> dict[str, float]:
    """blocked: tickers that may be exited but not (re-)entered this review,
    e.g. names stopped out earlier in the week. Gives stop-exits a minimum
    one-review cooldown instead of instant whipsaw re-entry."""
    blocked = blocked or set()
    ranked = sorted(convictions, key=convictions.get, reverse=True)
    rank = {t: i + 1 for i, t in enumerate(ranked)}

    # ---- exits ----
    survivors: dict[str, PositionState] = {}
    cooldown: set[str] = set()  # exited by stop/time-stop: no same-review re-entry
    for t, pos in positions.items():
        if t not in convictions:
            continue  # left the scored set -> exit
        r = rank[t]
        entry_ret = prices_today.get(t, pos.entry_price) / pos.entry_price - 1
        if r > cfg.exit_rank:
            continue  # conviction decayed
        if entry_ret <= -cfg.stop_loss:
            cooldown.add(t)
            continue  # stop-loss
        if (as_of - pos.entry_date).days > cfg.max_hold_days:
            cooldown.add(t)
            continue  # time-stop: thesis must work within ~4 months
        survivors[t] = pos

    # ---- entries ----
    no_enter = blocked | cooldown
    candidates = [
        t for t in ranked[: cfg.entry_rank]
        if t not in survivors and t not in no_enter
    ]
    slots = max(cfg.max_positions - len(survivors), 0)
    holdings = list(survivors) + candidates[:slots]
    if not holdings:
        return {}

    # ---- conviction-tilted weights (Jev "Choice"): tilt only redistributes ----
    base = min(1.0 / len(holdings), cfg.max_position_weight)
    tilts = {t: 0.75 + 0.5 * convictions[t] for t in holdings}
    mean_tilt = sum(tilts.values()) / len(tilts)
    targets = {t: base * tilts[t] / mean_tilt * regime_factor for t in holdings}

    # ---- sector cap: drop lowest-conviction names in overweight sectors ----
    def sector_w(sec: str) -> float:
        return sum(w for t, w in targets.items() if sectors.get(t) == sec)

    changed = True
    while changed:
        changed = False
        for sec in {sectors.get(t) for t in targets}:
            while sector_w(sec) > cfg.max_sector_weight and targets:
                drop = min(
                    (t for t in targets if sectors.get(t) == sec),
                    key=lambda t: convictions[t],
                )
                del targets[drop]
                changed = True

    return targets
