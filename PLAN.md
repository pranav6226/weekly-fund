# Weekly Quant Fund — Build Plan

Drafted 2026-09-24. Status: plan approved in principle, build kicks off tomorrow.

## 1. What this is (and isn't)

- A **personal systematic trading system**: Pranav's own capital, run by an agent swarm + Jev decision layer. Not a consumer app, not offered to others — no ETF labeling, no advisory registration questions.
- Strategy in one line: **weekend research, signal-driven holding.** Every weekend a research swarm scores a 200–500 stock universe; Jev turns research into a ranked portfolio; positions are held until the thesis breaks (typical 1–4 weeks), not force-liquidated on Friday.
- Validation: **paper trade on Alpaca for ~1 month**, then evaluate with real fills before any real capital.

## 2. Non-negotiables (from the Baskets post-mortem)

The Baskets AI audit found the replay engine clean but the methodology leaking. This build must:

1. **Walk-forward only.** Research at time T may use only data available at T. No optimizing weights on the full window and backtesting the same window (the `max_sharpe`-on-historical-means leak).
2. **Frictions modeled.** Commissions, spread/slippage, and short-term tax reality in every simulation. Weekly turnover lives or dies on costs.
3. **Rebalance actually wired.** (Baskets shipped a `rebalance_frequency` param that did nothing.)
4. **Decision audit log.** Every position records *why* it was taken (agent scores, Jev inputs) so we can learn which research agents actually add alpha.

## 3. Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ DATA LAYER  │────▶│ RESEARCH SWARM   │────▶│ DECISION LAYER  │────▶│  EXECUTION   │
│ (nightly +  │     │ (weekend batch)  │     │ (Jev + portfolio│     │ (Alpaca paper│
│  weekend)    │     │                  │     │  construction)  │     │  trading API)│
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
       ▲                     │                        │                      │
       │                     ▼                        ▼                      ▼
       │              ┌──────────────────────────────────────────────────────────┐
       └──────────────│ MONITORING AGENT (daily/intraday): stops, invalidation,  │
                      │ corporate actions, kill-switch ──▶ can flatten via Alpaca │
                      └──────────────────────────────────────────────────────────┘
```

### 3.1 Data layer
- **Prices/volumes:** Alpaca market data (reuse patterns from `baskets/backend/utils/alpaca_client.py`; note free tier = IEX feed, fine for paper). Nightly ingest into local DB (SQLite to start, Postgres if it grows) with the upsert-cache pattern — but **store as-of timestamps** so research queries are point-in-time.
- **Fundamentals/earnings calendar:** Finnhub (already a fallback in baskets) — earnings dates are critical: never open a new position into earnings within the expected hold window unless that's the explicit thesis.
- **News/sentiment:** TBD provider (Finnhub news, or a news API). Must be timestamped; sentiment agent only sees articles published before T.
- **Corporate actions:** splits/dividends adjusted at ingest; adjustment factor logged.

### 3.2 Research swarm (weekend batch, Grist-dispatchable)
Four (ish) parallel agents, each emits a **0–1 score per ticker + rationale**, using only data with `as_of <= T`:

| Agent | Inputs | Signal |
|---|---|---|
| Technicals / momentum | price/volume history | trend, momentum, volatility regime, mean-reversion |
| Fundamentals / earnings | financials, earnings dates, revisions | quality, earnings drift setup |
| News / sentiment | timestamped news | sentiment surprise, event detection |
| Macro / sector | sector ETFs, rates, VIX | regime filter, sector rotation tilt |

Each agent's output schema is fixed: `{ticker, score, confidence, rationale, as_of}`. Scores are **not** predictions of returns — they're research features.

### 3.3 Decision layer (Jev + portfolio construction)
- **Jev** takes the agent score vectors and produces calibrated **position convictions** (Score/Choice primitives — ranking under uncertainty, not price prediction). This is the confidence gate, same role it plays in Grist.
- **Portfolio construction** (deterministic, code not vibes):
  - Max positions: ~15–25; per-position cap (e.g. 8–10%); sector cap (e.g. 30%)
  - Entry: conviction above threshold AND passes risk filters (liquidity, no earnings collision)
  - Exit: conviction decayed below threshold, stop-loss hit, thesis invalidation flag from monitor, or a strictly better candidate needs the capital
  - Rebalance: weekly review; drift bands (e.g. ±25% from target weight) trigger trims, not full turnover

### 3.4 Execution (Alpaca paper)
- Paper trading API; Monday entries via market-on-open or limit orders; Alpaca handles fractional shares.
- Every order tagged with the decision-log ID for later attribution.
- Start: **no leverage, long-only.** (Shorting/married complexity comes after the paper month, if ever.)

### 3.5 Monitoring agent
Runs daily (intraday if we want it): checks stop-loss levels, scans for thesis-breaking news on held names, watches corporate actions, and owns a **kill-switch** (flatten on data outage, API errors, or drawdown circuit-breaker, e.g. −8% from equity peak pauses new entries).

### 3.6 Walk-forward simulator (build BEFORE live paper)
Event-driven, separate from the live loop:
- Steps through history week by week; at each step the *frozen* research+decision pipeline runs on as-of data only.
- Includes commissions + slippage model; reports net of costs.
- Purpose is pipeline validation (does the machinery work, are costs survivable), not proof of alpha — the paper month is the real test.

## 4. Build phases

- **Phase 0 — Data + broker plumbing.** Alpaca paper client, nightly ingest, as-of DB schema. Reuse baskets' Alpaca client patterns.
- **Phase 1 — Research swarm.** The four agents with fixed output schema, runnable as a weekend batch (Grist dispatch or cron).
- **Phase 2 — Decision + construction.** Jev integration (key stays in his `.env`), deterministic portfolio rules.
- **Phase 3 — Monitoring agent.** Stops, invalidation, kill-switch, Alpaca flatten path.
- **Phase 4 — Walk-forward simulator.** With frictions. Must pass before paper.
- **Phase 5 — Paper trade loop.** 1 month, weekly reviews with him, decision-attribution analysis at the end.

## 5. Open questions for tomorrow's kickoff

1. Universe: 200 or 500 names? (Start smaller — 200 liquid large/mid-caps keeps data costs and noise down.)
2. Paper capital size + per-position risk cap.
3. Alpaca tier: paper is free; confirm data subscription covers what the swarm needs.
4. Jev access: key lives in his `.env`, never in chat (standing rule).
5. Grist as the swarm runner vs. plain cron jobs on this VM.

## 6. What we are NOT doing

- No consumer app, no "create your own ETF" UX, no marketing of returns.
- No real capital until the paper month is reviewed together.
- No in-sample optimization anywhere in the pipeline, ever.
