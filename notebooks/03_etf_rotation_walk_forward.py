"""Walk-forward validation of ETF rotation strategy.

Setup:
- 3-year train / 1-year test windows, stepping 1 year at a time
- Each test window starts with a fresh ₹10k
- Same fixed strategy parameters across all windows (no per-window refit)

This isolates regime-robustness: if a window's Sharpe is materially below
the in-sample 1.33, the strategy's edge is regime-dependent.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig
from algo.backtest.metrics import compute_metrics
from algo.backtest.walk_forward import (
    make_windows,
    run_walk_forward,
    summarize_windows,
)
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams

OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Fetching ETF prices (max history)...")
    symbols = to_yfinance(ETF_ROTATION)
    # Use a long history so we get multiple windows
    prices = adjusted_close_panel(symbols, start="2014-01-01", end=date.today().isoformat())
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    prices = prices.dropna(how="all")
    print(f"  history: {prices.index.min().date()} -> {prices.index.max().date()}, "
          f"{len(prices)} trading days")

    strat = EtfRotation(params=EtfRotationParams())
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        product=Product.CNC,
        slippage_bps=5.0,
        min_position_inr=2_000.0,
    )
    windows = make_windows(prices.index, train_years=3, test_years=1, step_years=1)
    print(f"  windows: {len(windows)}")
    for w in windows:
        print(f"    train {w.train_start.date()}->{w.train_end.date()}  "
              f"test {w.test_start.date()}->{w.test_end.date()}")

    print("\nRunning walk-forward...")
    results = run_walk_forward(
        prices=prices,
        signal_fn=strat.signals,
        config=cfg,
        windows=windows,
        warmup_days=strat.params.trend_sma,
    )

    summary = summarize_windows(results)
    print("\n=== Per-window OOS results ===")
    print(summary.to_string(index=False))

    # Aggregate OOS equity into a single curve (chain returns; reset capital each window)
    print("\n=== Aggregate OOS metrics ===")
    chained_returns = []
    for wr in results:
        rets = wr.result.equity.pct_change().dropna()
        chained_returns.append(rets)
    if chained_returns:
        oos_rets = pd.concat(chained_returns).sort_index()
        oos_equity = (1 + oos_rets).cumprod() * cfg.initial_capital
        agg = compute_metrics(oos_equity)
        for k in ("cagr_pct", "sharpe", "sortino", "max_drawdown_pct", "calmar"):
            print(f"  {k:24s} {getattr(agg, k):>10.3f}")

        # In-sample comparison reference (from notebook 02)
        print("\nIn-sample 5y reference (notebook 02): CAGR 23.55, Sharpe 1.33, MaxDD -24.40")

        oos_equity.to_frame("equity").to_parquet(
            OUT_DIR / "etf_rotation_walkforward_equity.parquet"
        )
        summary.to_parquet(OUT_DIR / "etf_rotation_walkforward_summary.parquet")
        print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
