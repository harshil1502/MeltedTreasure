"""Run the swing momentum strategy on Nifty 50, 5-year backtest.

Usage:
    PYTHONPATH=src python notebooks/01_swing_momentum_backtest.py

Output:
    - Console: metrics table + cost summary
    - data/cache/swing_momentum_equity.parquet (equity curve)
    - data/cache/swing_momentum_trades.parquet (trade log)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig, run_backtest
from algo.backtest.metrics import compute_metrics
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import NIFTY_50, to_yfinance
from algo.strategies.swing_momentum import SwingMomentum, SwingMomentumParams

OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Fetching 5y of Nifty 50 daily prices from yfinance...")
    symbols = to_yfinance(NIFTY_50)
    prices = adjusted_close_panel(symbols, start="2020-01-01", end=date.today().isoformat())
    # Strip the .NS suffix for cleanliness
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    # Drop columns with too many NaNs (recent listings, missing data)
    coverage = prices.notna().sum() / len(prices)
    keep = coverage[coverage > 0.9].index
    prices = prices[keep].dropna(how="all")
    print(f"  {len(prices)} trading days × {len(prices.columns)} symbols after coverage filter")

    print("\nGenerating swing momentum signals (lookback=20, sma=50, top_n=2, weekly Mon)...")
    strat = SwingMomentum(params=SwingMomentumParams(lookback=20, trend_sma=50, top_n=2))
    weights = strat.signals(prices)
    print(f"  {len(weights)} rebalance days, "
          f"avg {weights.gt(0).sum(axis=1).mean():.2f} positions held")

    print("\nRunning backtest (₹10,000 CNC, 5 bps slippage)...")
    result = run_backtest(
        prices=prices,
        target_weights=weights,
        config=BacktestConfig(
            initial_capital=10_000.0,
            product=Product.CNC,
            slippage_bps=5.0,
            min_position_inr=2_000.0,
        ),
    )

    metrics = compute_metrics(result.equity, result.trades)
    print("\n=== Performance ===")
    for k, v in metrics.as_dict().items():
        print(f"  {k:24s} {v:>10.3f}")

    print("\n=== Cost summary ===")
    if not result.trades.empty:
        cost_breakdown = result.trades[
            ["cost_brokerage", "cost_stt", "cost_exchange", "cost_sebi",
             "cost_stamp", "cost_gst", "cost_dp", "cost_slippage"]
        ].sum()
        print(cost_breakdown.to_string())
        print(f"  total cost ₹: {result.cost_total:.2f}")
        print(f"  cost as % of starting capital: "
              f"{result.cost_total / result.config.initial_capital * 100:.2f}%")

    # Compare against buy-and-hold NIFTYBEES proxy: equal-weight Nifty 50 for now
    print("\n=== Benchmark (equal-weight buy-and-hold) ===")
    benchmark_returns = prices.pct_change().mean(axis=1)
    bh_equity = (1 + benchmark_returns).cumprod() * result.config.initial_capital
    bh_metrics = compute_metrics(bh_equity.dropna())
    print(f"  CAGR %                   {bh_metrics.cagr_pct:>10.3f}")
    print(f"  Sharpe                   {bh_metrics.sharpe:>10.3f}")
    print(f"  Max DD %                 {bh_metrics.max_drawdown_pct:>10.3f}")

    result.equity.to_frame("equity").to_parquet(OUT_DIR / "swing_momentum_equity.parquet")
    if not result.trades.empty:
        result.trades.to_parquet(OUT_DIR / "swing_momentum_trades.parquet")
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
