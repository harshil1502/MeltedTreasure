# Options Strategy Validation Gauntlet

**Purpose:** establish a forced-path that any options strategy must pass before
real capital is deployed. Every stage has explicit pass criteria. Failing any
stage means the strategy does not graduate. No exceptions, no "almost passed."

This document exists because:
1. Synthetic-pricing backtests can flatter strategies by 20-50% (no vol crush,
   no real bid-ask, no liquidity gaps).
2. SEBI publishes that ~89% of retail F&O traders lose money. Default prior is
   "this strategy doesn't work." High bar to overturn.
3. The current ORB synthetic backtest (n=28, +274% in 60d) is **NOT** evidence
   of edge. One trade was 60% of PnL. Sample size alone disqualifies it.

---

## Stage 0: Strategy specification (no capital required)

Before anything else: write down the exact rules.

- [ ] Entry conditions, in code, deterministic
- [ ] Exit conditions: stops, targets, time-stops, all explicit
- [ ] Position sizing rule (lots per ₹X capital)
- [ ] Maximum daily / weekly drawdown after which the strategy pauses
- [ ] One-page strategy memo: what edge does this exploit, why does it exist,
      who is on the other side of the trade?

**Pass criterion:** The strategy can be executed by reading the code, with no
human discretion required.

---

## Stage 1: Real-data backtest (₹2k Kite Connect, ~1 month)

- [ ] Sign up for Kite Connect (kite.trade): ₹2k/month
- [ ] Pull 12 months of intraday Nifty options data for at least 5 strikes
      around ATM, plus index minute bars
- [ ] Run the strategy with **real** entry/exit prices (not BSM theoretical)
- [ ] Apply realistic slippage: 0.25% for ATM, 0.75% for OTM weeklies
- [ ] Apply the F&O cost model at every leg

### Pass criteria (ALL must hold)

| # | Metric | Bar |
|---|---|---|
| 1 | Trade count | n ≥ 150 |
| 2 | Win rate | ≥ 40% (for buy-side) or ≥ 65% (for sell-side) |
| 3 | Largest single trade as % of total PnL | ≤ 15% |
| 4 | Maximum drawdown | ≤ 25% of starting equity |
| 5 | Net PnL after costs | > 0 in at least 8 of 12 calendar months |
| 6 | Sharpe (annualized, daily returns) | ≥ 1.0 |
| 7 | Avg loss as multiple of avg win | sell-side: ≤ 4x; buy-side: ≤ 2x |

**If any criterion fails: strategy is rejected.** Do not "tune until it passes" —
that's overfitting. Either it works on the first honest run or it doesn't.

---

## Stage 2: Walk-forward out-of-sample test

Even after Stage 1 passes, the strategy might be in-sample-overfit. Walk-forward
proves the rules generalize.

- [ ] Split the 12 months into 9-month train + 3-month test
- [ ] Lock all parameters from train; do not retune for test
- [ ] Run on test period

### Pass criteria

- [ ] Out-of-sample Sharpe ≥ 70% of in-sample Sharpe
- [ ] Out-of-sample win rate within 8 percentage points of in-sample
- [ ] Out-of-sample max drawdown not more than 1.5x in-sample max DD

**If degradation is severe: the in-sample edge was overfit. Reject.**

---

## Stage 3: Paper trading (live data, ~2 months)

Real data, real timestamps, but no money risked. Confirms execution logic
matches backtest assumptions.

- [ ] Build paper-execution layer that reads live Kite ticks and "fills"
      orders at next-print-after-signal
- [ ] Run for 40+ trading days
- [ ] Log every signal, fill, and PnL

### Pass criteria

- [ ] Realised paper PnL/trade within ±20% of backtest PnL/trade
- [ ] Slippage observed ≤ slippage assumed in backtest
- [ ] No execution edge cases unaccounted for (gaps, halts, expiry-day pin)

---

## Stage 4: Tiny live deployment (₹50k or 1 lot, ~1.5 months)

The first time real money is risked. Size below the level where a -100%
outcome materially affects the operator. ₹50k or one minimum-lot position.

- [ ] Open Kite trading account (if not already)
- [ ] Deploy the strategy with the smallest viable position size
- [ ] Run for ≥ 30 trades
- [ ] Compare live execution vs paper trading

### Pass criteria

- [ ] Live PnL/trade within ±25% of paper trading PnL/trade
- [ ] No surprises in execution: no rejected orders, no margin calls, no
      unexpected interactions with broker risk-management systems
- [ ] Operator psychology: did you actually follow the rules, or did you
      override the system at any point? If yes → this is a process failure,
      not a strategy failure, but it must be addressed before scaling.

---

## Stage 5: Scale to full size

Only after Stages 1-4 all pass.

- [ ] Scale to target capital (₹1L–₹5L per the current plan)
- [ ] Continuous monitoring: daily PnL, drawdown, parameter drift
- [ ] Pre-defined kill switch: if rolling 30-day PnL falls below -15% of
      deployed capital, pause and re-validate

---

## Total time-to-deployment

| Stage | Duration | Cost |
|---|---|---|
| 0    | 1 week | 0 |
| 1    | 2 weeks | ₹2k (Kite) |
| 2    | 1 week | included |
| 3    | 8 weeks | ₹4k (Kite × 2) |
| 4    | 6 weeks | ₹4k (Kite × 2) + small live capital |
| 5+   | indefinite | ongoing |
| **Total to live scaling** | **~5 months** | **~₹10k infra + capital** |

---

## What CANNOT be skipped

- Real options data. BSM theoretical prices systematically overstate winning
  trades' premiums (no vol crush) and understate slippage on OTM strikes.
- Walk-forward. In-sample success is meaningless without it.
- Paper trading. Backtests don't reveal execution edge cases.

---

## Current state (as of 2026-05-07)

- [x] Stage 0 partially done: ORB strategy specified in
      `src/algo/strategies/options_orb.py`
- [ ] Stage 1: blocked on Kite Connect signup
- [ ] Stages 2-5: gated on Stage 1

## Next concrete action for the operator

Sign up for Kite Connect at https://kite.trade. Provide API key + access token,
or download a 12-month historical options data dump and place it under
`data/kite/`. Until then, all further synthetic backtesting is theatre.
