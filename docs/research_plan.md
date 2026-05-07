# Research Plan

## Hypotheses

### H1 — Swing momentum on Nifty 50 has positive expected value after costs

Cross-sectional and time-series momentum on Indian large-caps has well-documented
persistence (Jegadeesh-Titman holds in EM equities; published evidence on NSE since 2005).
With 1–10 day holds, brokerage drag is amortized and STT/stamp impact is bounded.

- **Universe**: Nifty 50 constituents (rolling, point-in-time)
- **Signal**: 20-day return rank, filtered by 50-SMA trend and ADX(14) > 20
- **Sizing**: equal-weight top-N, N ∈ {1, 2, 3} given ₹10k constraint
- **Holding**: 5 trading days, exit on stop or signal flip
- **Stop**: ATR(14) × 2 below entry
- **Success criterion**: Sharpe > 1.0, max DD < 15%, after-cost CAGR > 12% on 5-year backtest

### H2 — Monthly ETF rotation reduces drawdown vs single-asset hold

Rotating across asset-class ETFs (equity / debt-proxy / gold) based on 3-month momentum
captures regime shifts. Zero delivery brokerage on Zerodha makes ETF rotation
cost-favorable.

- **Universe**: NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES
- **Signal**: rank by 3-month total return; hold top-1 or top-2
- **Rebalance**: monthly (last trading day)
- **Success criterion**: Sharpe > 0.8, max DD < 12%, beats 60/40 NIFTYBEES/LIQUIDBEES blend

### H3 — Intraday ORB on liquid Nifty 50 names cannot overcome costs at ₹10k

Documenting expected failure. Five-minute opening-range breakout on top-10 Nifty by
ADV. Expected to lose money after costs at this account size; serves as a calibration
point for the cost model and demonstrates why intraday is gated until capital scales.

- **Universe**: top 10 Nifty 50 by 30-day ADV (RELIANCE, HDFCBANK, etc.)
- **Signal**: break of first 15-minute high/low with volume confirmation
- **Stop**: opposite end of opening range
- **Target**: 1.5× range, exit by 15:15 IST otherwise
- **Success criterion (negative)**: Confirm cost drag > gross edge on 2-year backtest

## Data requirements

| Strategy | Frequency | Source | Coverage |
|----------|-----------|--------|----------|
| H1 | Daily OHLCV | yfinance + NSE bhavcopy | 2019-01 to today |
| H2 | Daily OHLCV | yfinance | 2019-01 to today |
| H3 | 5-min bars | Kite historical (Phase 2) | last 60 days for prelim |

## Cost model assumptions (Zerodha, retail)

### Equity intraday (MIS)
- Brokerage: min(0.03% × turnover, ₹20) per leg
- STT: 0.025% on sell side
- Exchange transaction: 0.00322% (NSE) per leg
- SEBI charges: 0.0001% per leg
- Stamp duty: 0.003% on buy side
- GST: 18% × (brokerage + exchange + SEBI)
- **Slippage assumption**: 5 bps each side on Nifty 50 large-caps; 15 bps on mid-cap

### Equity delivery (CNC)
- Brokerage: ₹0
- STT: 0.1% per leg
- Exchange / SEBI: same as above
- Stamp duty: 0.015% on buy
- GST: 18% × (exchange + SEBI)
- **DP charges**: ₹13.5 + 18% GST ≈ ₹15.93 per scrip per day on sell — **critical at small size**

## Validation methodology

1. **Walk-forward only** — no in-sample/out-of-sample peeking. 3-year train, 1-year test, roll annually.
2. **Point-in-time universe** — survivorship bias kills momentum backtests; use historical Nifty 50 constituents.
3. **Conservative slippage** — bake in 1.5× the model-default slippage as a stress test.
4. **DP charges + STT booked correctly** — common retail backtest error to omit.
5. **Position-sizing realism** — at ₹10k, you can't buy 1 share of MRP > ₹10k stocks; whole-share constraint matters.

## Decision gate

Strategies that fail to clear after-cost Sharpe > 0.7 on walk-forward go to the
recycle bin. No live capital is committed without a strategy passing the gate
**twice** — once on a 5-year backtest and once on 6-month forward paper.
