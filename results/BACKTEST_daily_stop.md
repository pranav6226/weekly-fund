# BACKTEST — daily stop enforcement vs weekly-only stops (A/B)

**Run:** 2026-10-15 (driver: scripts/backtest_daily_stop.py; raw output: results/backtest_daily_stop/metrics.json)
**Window:** 2020-01-01 → 2026-09-25 · **Universe:** 503 tickers · **Reviews:** 351 weekly
**Mandate:** core-4desk — technicals 0.40 / fundamentals 0.30 / filings 0.15 / macro 0.15
**Question:** the v3 simulator has always enforced the 10% stop on daily closes; the live loop checked stops only at the Saturday review. What is that gap worth?

Both variants share one memoized scoring pass and identical data; the only difference is the simulator's `daily_stop` flag.

| metric | A: daily stops (v3 as validated) | B: weekly-only (live behavior at the time) |
|---|---|---|
| total return | +331.2% | +290.6% |
| CAGR | 24.3% | 22.5% |
| Sharpe | 1.23 | 1.13 |
| max drawdown | −26.8% | −30.7% |
| trades | 4,736 | 4,553 |
| round trips | 406 | 407 |
| win rate | 49.8% | 49.6% |
| turnover | 303.5× | 263.1× |
| mid-week stop exits | 179 | 0 |
| final equity ($100K start) | $431,211 | $390,585 |

## Read

Daily enforcement wins on return AND risk: +1.8pp CAGR and ~4pp shallower max drawdown, for ~15% more turnover — already net of modeled slippage. Win rate is unchanged; stops cut losers shorter rather than picking better. ~179 mid-week exits over 351 weeks ≈ one a fortnight.

## Caveats

- Variant A does not exactly reproduce the published v3 headline (+273.7% / 21.7% CAGR): the price panel was refreshed/restated after the September v3 run. Both variants here ran on the same current panel, so the DELTA is the clean read; the absolute levels carry the standing v3 caveats (in-sample momentum window, survivorship bias, tax unmodeled).
- Decision (2026-10-16, operator-approved): build daily stop enforcement into the live worker — see scripts/stop_check.py, scheduled Mon–Fri 13:35 PT by scripts/autonomous_worker.py.
