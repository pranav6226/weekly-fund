# Weekly Fund

Personal systematic trading fund. Weekly research cadence, signal-driven
exits — not a consumer app, not offered to anyone else.

Pipeline: **data → research swarm → Jev decision layer → portfolio
construction → Alpaca execution**, with a monitoring agent holding the
kill-switch. Full architecture: [`PLAN.md`](PLAN.md).

## Status

- [x] Phase 0 — data plumbing (yfinance ingest → SQLite; Alpaca/Finnhub
      clients ported from the Baskets codebase for the live loop)
- [x] Phase 1 — research agents (v1: deterministic momentum + low-vol;
      LLM swarm plugs into the same `score_universe()` signature)
- [x] Phase 2 — decision layer (`construct_targets()` shared by backtest
      and live; Jev interface stubbed in `decision/jev.py`)
- [x] Phase 4 — walk-forward simulator with frictions (built before any
      paper trading, per the Baskets post-mortem)
- [ ] Phase 3 — monitoring agent
- [ ] Phase 5 — Alpaca paper-trade loop (~1 month, then review)

## Quickstart

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 1. download history (S&P 500, 2018->today, total-return adjusted)
./.venv/bin/python scripts/download_history.py

# 2. run the walk-forward backtest
./.venv/bin/python scripts/run_backtest.py --start 2020-01-01 --end 2026-09-25

# 3. unit tests for the portfolio constructor
./.venv/bin/python tests/test_portfolio.py
```

## Methodology notes (read before trusting any number)

- **Walk-forward only.** Every review uses data with `date <= T`. No
  in-sample optimization: rules are fixed policy, not fit on the window.
- **Frictions.** 5 bps/side slippage against the open on every fill;
  zero commission (Alpaca). Taxes not modeled.
- **Known optimistic biases in v1:** the universe is *current* S&P 500
  constituents (survivorship bias — delisted names can't be picked);
  prices are total-return adjusted (correct) but corporate-action edge
  cases aren't separately validated.
- **v1 strategy is a machinery test.** Momentum + low-vol is a baseline
  to validate data → scores → portfolio → fills → accounting. The edge,
  if any, comes later from the research swarm + Jev.

## Layout

```
src/fund/
  config.py            fund-wide policy (shared by backtest + live)
  data/
    clients/alpaca.py  ported from Baskets (live loop)
    clients/finnhub.py ported from Baskets (live loop)
    db.py / store.py   SQLite market-data store
  research/
    agents.py          v1 heuristic agents; swarm plugs in here
    schema.py          fixed AgentScore output contract
  decision/
    portfolio.py       THE shared constructor (backtest == live logic)
    jev.py             Jev interface (key in user .env; live loop only)
  backtest/simulator.py  walk-forward weekly simulator with frictions
  execution/broker.py    Alpaca paper wrapper (Phase 5)
  monitor/monitor.py     stops / invalidation / kill-switch (Phase 3)
scripts/
  download_history.py  yfinance -> SQLite
  run_backtest.py      CLI: simulate, compare vs SPY, write results/
```
