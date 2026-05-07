# MeltedTreasure — Trading Algorithm R&D

R&D for an algorithmic trading system targeting the Indian equity market via Zerodha,
starting with ₹10,000 of capital.

## Capital reality check

₹10k is below the practical viability threshold for most algo strategies. Cost floor on
intraday equity is ~0.5–0.6% of capital per round-trip (brokerage + STT + GST + exchange
+ stamp + slippage). Strategies must clear ~0.7%+ edge per trade after costs.

The ₹10k is treated as a **live-validation budget after backtesting** — not a serious
trading account. The product being built is the pipeline (data → signal → backtest →
execution → monitoring); capital is a config knob.

## R&D phases

| Phase | Goal | Deliverable |
|------:|------|-------------|
| 0 | Infrastructure | Data loaders, accurate cost model, universe defs, backtest harness |
| 1 | Strategy research | Backtests for 3 candidates: swing momentum, ETF rotation, intraday ORB |
| 2 | Paper trading | Selected winner runs on live Kite ticks, no real orders |
| 3 | Live (₹10k) | Semi-automated execution after SEBI compliance review |

Currently in **Phase 0**.

## Strategy candidates (Phase 1)

1. **Swing momentum on Nifty 50** — 1–10 day holds, 20/50 SMA + ADX filter
2. **ETF rotation** — monthly momentum rotation across NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES
3. **Intraday opening-range breakout** — Nifty 50 large-caps, included as a baseline (expected
   to fail cost-edge test at ₹10k; documenting the failure is part of the research)

## Stack

- Python 3.11+, managed via `uv` or `pip`
- `pandas` / `numpy` for data
- `vectorbt` for vectorized backtesting
- `yfinance` + NSE bhavcopy for free daily data (Phase 0–1)
- `kiteconnect` for live data and execution (Phase 2+, ₹2k/mo)
- Jupyter for exploratory research; `src/algo/` for production-bound code

## Compliance

SEBI's Sept 2024 framework on retail algorithmic trading requires broker approval and
unique strategy IDs for auto-executed orders. System will be designed semi-automated
(signal → manual confirm) for the ₹10k phase to stay clean, with full automation
gated behind broker sign-off.

## Layout

```
src/algo/
  data/        # NSE / yfinance loaders, universe definitions
  backtest/    # cost model, engine wrappers
  strategies/  # strategy implementations
  risk/        # position sizing, drawdown control
notebooks/     # Jupyter research notebooks
docs/          # research plan, strategy notes
tests/         # unit tests (cost model, sizing)
data/          # local cache (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
jupyter lab
```
