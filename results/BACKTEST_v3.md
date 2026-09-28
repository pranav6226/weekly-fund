# BACKTEST v3 — 4-desk swarm on the free stack

**Run:** 2026-09-28 00:13 PDT (run5; runs 1–4 killed: two for perf fixes, one infra session reap)
**Window:** 2020-01-01 → 2026-09-25 · **Universe:** 503 tickers · **Reviews:** 351 weekly
**Mandate:** core-4desk — technicals 0.40 / fundamentals 0.30 / filings 0.15 / macro 0.15
**Data:** 100% free — SEC EDGAR (XBRL companyfacts + 8-K FTS) + FRED; no Financial Datasets
**Frictions:** modeled (walk-forward, non-negotiable)

## Headline

| metric | fund-v3 | SPY buy-and-hold |
|---|---|---|
| total return | **+273.7%** | +159.7% |
| CAGR | **21.7%** | 15.3% |
| Sharpe | **1.125** | 0.81 |
| max drawdown | **−27.8%** | −33.7% |
| final equity ($100k start) | **$373,671** | $259,716 |

Trades: 4,718 (410 round trips) · win rate 51% · turnover 287.9×

## Read

- v3 beats SPY on return, risk-adjusted return, and drawdown depth. The regime
  layer did its job: risk-off through 2022 (trend-broken + vol-spike) and the
  April-2025 stress print, neutral into late 2026 on weak breadth.
- v3 trails v1 (40.0% CAGR) and v2.1 (31.7%) — expected. Those ran on the paid
  Financial Datasets feed with richer fundamentals; v3 is the free-stack build.
  The gap is the price of $0/month data, and it's still 6.4pp/year over SPY.
- Turnover is the thing to watch: 287.9× over 6.7 years with 4,718 trades.
  Frictions are modeled, so the 21.7% is net — but live slippage on a weekly
  rebalance of this size is the main paper-trading question.

## Honest caveats (unchanged)

- In-sample on a historic momentum regime; survivorship bias (current
  constituents backfilled); short-term tax unmodeled; FRED revision lookahead
  is a minor caveat (macro desk is 0.15 weight, blended 50/50 with price
  regime). The paper-trading month is the real test.
- Desk weights are documented priors, not fit — discipline held from v2.1.

## Perf notes (for the build log)

- Full 351-review run: 1081.5s (~18 min) after three fixes —
  (1) compact `annual_facts` memo instead of raw 10MB JSONs (the raw memo
  OOM-killed the box); (2) `filing_events` parses dates once via
  `date.fromisoformat` (per-event `pd.to_datetime` was 87% of a review);
  (3) string-slice `.loc` price lookups and ISO string comparison in the
  macro desk instead of per-call `pd.to_datetime`.
- Infra: one run died at review ~226 with no traceback — the VM reaped the
  background session, not a strategy bug (probe of reviews 226–230 all pass).
  Relaunched detached via `setsid`/`nohup`.

## Next

- Phase 3 monitor + Phase 5 Alpaca paper loop (unchanged).
- His lanes: `FRED_API_KEY` in `.env`; GitHub repo + push; transient Jev key
  when ready. Open kickoff questions: universe size, paper capital,
  per-position risk cap, Alpaca tier.
