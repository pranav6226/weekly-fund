# Backtest v1 — momentum + low-vol baseline (2026-09-27)

## What was tested
Not a claim about the fund's future edge — a **machinery validation**:
data → scores → portfolio construction → fills → accounting, all
walk-forward with frictions.

- **Universe:** S&P 500, current constituents (503 tickers)
- **Signal:** 126-day momentum skipping the most recent 21 days (70%) +
  low 63-day volatility (30%), z-scored cross-sectionally, weekly review
- **Portfolio:** top-20 by score, equal weight (8% cap), 30% sector cap,
  10% stop-loss, hysteresis exit band (rank > 30)
- **Execution:** orders Friday close → filled next open ± 5 bps/side,
  $0 commission, $500 no-trade band
- **Window:** 2020-01-01 → 2026-09-25 (351 weekly reviews, 4,194 trades)

## Results (net of costs)

| metric | fund v1 | SPY buy-and-hold |
|---|---|---|
| total return | +857% | +160% |
| CAGR | 40.0% | 15.3% |
| Sharpe | 1.54 | 0.81 |
| max drawdown | -30.4% | -33.7% |
| turnover | 367x (6.75y) | — |
| round-trip win rate | 60.9% (322) | — |

Year by year: 2020 +64% (+17% SPY), 2021 +27% (+31%), 2022 **-3%**
(-19%), 2023 +44% (+27%), 2024 +53% (+26%), 2025 +33% (+18%),
2026 YTD +58% (+13%).

## Why it's this high (honest decomposition)
1. **Regime.** 2020–2026 was a historic momentum regime: COVID rebound,
   then the AI buildout. The strategy rode TSLA/NVDA/AMD (2020),
   dodged the 2022 drawdown by rotating out, then rode SMCI/APP/CVNA/
   MU/BE/HOOD (2023–2026). Momentum concentrates in exactly these names.
2. **Survivorship bias.** The universe is *today's* S&P 500 — dead names
   can't be picked. Biases results up by an unmeasured low-single-digit
   %/yr.
3. **In-sample strategy selection.** The signal family (12-1 momentum)
   is literature-standard, not fit to this window — but we chose to test
   *momentum*, knowing it works. Do not annualize 40% into an expectation.

## Why we trust the simulator anyway
A **seeded random scorer** run through the identical pipeline returned
+91% total / 10.1% CAGR / 0.85 Sharpe — *below* buy-and-hold, exactly as
a churny random 20-stock portfolio should. If the accounting leaked
future returns, random would print fantasy numbers too. It doesn't.

Plus: unit tests on the portfolio constructor (5/5), warmup/full-history
slicing verified, Friday→Monday execution matches the live-loop design.

## What v1 does NOT model
Short-term capital-gains tax (this turnover would be brutal), market
impact beyond 5 bps, or delisted-name survivorship. All three cut the
real number — the paper-trading month is the honest test.
