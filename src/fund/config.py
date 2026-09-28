"""Fund-wide configuration. One object, shared by the backtester and the live loop.

Nothing here is fit on historical data -- these are policy choices, so there
is no in-sample leak from tuning them (though any *change* made after seeing
backtest results must be disclosed as a research decision, not a free lunch).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FundConfig:
    # --- universe ---
    universe_csv: str = "data/universe.csv"
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0  # 63-day median, USD

    # --- portfolio construction ---
    initial_capital: float = 100_000.0
    max_positions: int = 20
    max_position_weight: float = 0.08
    max_sector_weight: float = 0.30
    entry_rank: int = 20   # hold roughly the top-N by score
    exit_rank: int = 30    # hysteresis band: keep while rank <= exit_rank
    stop_loss: float = 0.10  # exit if close falls 10% below entry

    # --- execution / frictions ---
    slippage_bps: float = 5.0      # per side, applied against the open
    commission_per_trade: float = 0.0  # Alpaca is commission-free
    min_trade_value: float = 500.0  # no-trade band: skip rebalances smaller than this

    # --- backtest window ---
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2026-09-25"
    warmup_days: int = 200  # trading days of history before the first review

    # --- research weights (v1 deterministic agents) ---
    w_momentum: float = 0.7
    w_lowvol: float = 0.3
    momentum_lookback: int = 126  # ~6 months
    momentum_skip: int = 21       # skip most recent month (12-1 style)
    vol_window: int = 63

    db_path: str = "data/market.db"
    results_dir: str = "results"
