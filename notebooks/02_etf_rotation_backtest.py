"""ETF rotation backtest, 5-year horizon.

Universe: NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES.
Strategy: monthly top-1 by 3-month momentum, with 200-SMA risk-off to LIQUIDBEES.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig, run_backtest
from algo.backtest.metrics import compute_metrics
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams

OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Fetching ETF prices from yfinance (5y)...")
    symbols = to_yfinance(ETF_ROTATION)
    prices = adjusted_close_panel(symbols, start="2019-01-01", end=date.today().isoformat())
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    coverage = prices.notna().sum() / len(prices)
    print(f"  coverage: {coverage.to_dict()}")
    prices = prices.dropna(how="all")

    strat = EtfRotation(params=EtfRotationParams(momentum_lookback=63, trend_sma=200, top_n=1))
    weights = strat.signals(prices)
    print(f"\nGenerated {len(weights)} monthly rebalance signals")
    chosen = weights.idxmax(axis=1)
    print(f"Allocation distribution:")
    print(chosen.value_counts())

    print("\nRunning backtest (₹10k CNC, 5 bps slippage)...")
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
        print(f"  total cost ₹: {result.cost_total:.2f}")
        print(f"  cost as % of starting capital: "
              f"{result.cost_total / result.config.initial_capital * 100:.2f}%")
        print(f"  trades: {len(result.trades)}  "
              f"(buys: {(result.trades['side']=='BUY').sum()}, "
              f"sells: {(result.trades['side']=='SELL').sum()})")

    # Benchmark: 100% buy-and-hold NIFTYBEES
    print("\n=== Benchmark: buy-and-hold NIFTYBEES ===")
    if "NIFTYBEES" in prices.columns:
        bh = prices["NIFTYBEES"].dropna()
        bh_eq = bh / bh.iloc[0] * 10_000.0
        bh_metrics = compute_metrics(bh_eq)
        for k in ("cagr_pct", "sharpe", "max_drawdown_pct"):
            print(f"  {k:24s} {getattr(bh_metrics, k):>10.3f}")

    result.equity.to_frame("equity").to_parquet(OUT_DIR / "etf_rotation_equity.parquet")
    if not result.trades.empty:
        result.trades.to_parquet(OUT_DIR / "etf_rotation_trades.parquet")
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
