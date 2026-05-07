"""Diagnose why lookback=42 and lookback=126 produce -99% drawdowns.

Hypothesis: yfinance adjusted-close on Indian ETFs has corporate-action
discontinuities (large single-day jumps) that the strategy reacts to. The
benchmark NIFTYBEES already showed a -90% DD that's impossible for
buy-and-hold — same data quality issue likely poisons signals at certain
lookbacks.

Diagnostics:
  1. Per-window OOS DDs at the failing lookbacks
  2. Scan all four ETFs for daily returns > 30% or < -30% (red flags)
  3. If found, mask and re-run the failing parameter set
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig
from algo.backtest.metrics import compute_metrics
from algo.backtest.walk_forward import make_windows, run_walk_forward, summarize_windows
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams


def main() -> None:
    print("Loading ETF prices (raw, no cleaning)...")
    symbols = to_yfinance(ETF_ROTATION)
    prices = adjusted_close_panel(symbols, start="2014-01-01", end=date.today().isoformat())
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    prices = prices.dropna(how="all")

    # --- 1. Scan for impossible daily returns ---
    print("\n=== 1. Scanning for daily-return outliers (|r| > 30%) ===")
    daily_rets = prices.pct_change()
    for col in daily_rets.columns:
        outliers = daily_rets[col][daily_rets[col].abs() > 0.30].dropna()
        if not outliers.empty:
            print(f"\n  {col}: {len(outliers)} outlier(s)")
            for d, r in outliers.items():
                px_before = prices[col].shift(1).loc[d]
                px_after = prices[col].loc[d]
                print(f"    {d.date()}: return {r*100:+.1f}%  (price {px_before:.2f} -> {px_after:.2f})")
        else:
            print(f"  {col}: clean")

    # --- 2. Per-window OOS at the failing lookbacks ---
    windows = make_windows(prices.index, train_years=3, test_years=1, step_years=1)
    cfg = BacktestConfig(initial_capital=10_000.0, product=Product.CNC, slippage_bps=10.0)

    for lookback in (42, 63, 126):
        print(f"\n=== 2. Per-window OOS at lookback={lookback}, sma=200 ===")
        strat = EtfRotation(params=EtfRotationParams(momentum_lookback=lookback, trend_sma=200))
        results = run_walk_forward(
            prices=prices, signal_fn=strat.signals, config=cfg,
            windows=windows, warmup_days=200,
        )
        summary = summarize_windows(results)
        print(summary.to_string(index=False))
        worst = summary["max_dd_pct"].min()
        print(f"  Worst single-window DD: {worst:.2f}%")

    # --- 3. If outliers found, mask them and re-run ---
    big_jumps = daily_rets.abs() > 0.30
    if big_jumps.any().any():
        print("\n=== 3. Masking outliers and re-running lookback=42 and 126 ===")
        clean_prices = prices.copy()
        # For each outlier day, replace with previous day's price (effectively zero return)
        for col in clean_prices.columns:
            mask = big_jumps[col].fillna(False)
            if mask.any():
                clean_prices.loc[mask, col] = pd.NA
        clean_prices = clean_prices.ffill().dropna(how="all")

        for lookback in (42, 126):
            strat = EtfRotation(params=EtfRotationParams(momentum_lookback=lookback, trend_sma=200))
            results = run_walk_forward(
                prices=clean_prices, signal_fn=strat.signals, config=cfg,
                windows=windows, warmup_days=200,
            )
            if not results:
                continue
            rets = pd.concat([r.result.equity.pct_change().dropna() for r in results]).sort_index()
            eq = (1 + rets).cumprod() * cfg.initial_capital
            m = compute_metrics(eq)
            print(f"\nlookback={lookback}, sma=200, prices cleaned:")
            print(f"  CAGR: {m.cagr_pct:.2f}%   Sharpe: {m.sharpe:.3f}   "
                  f"MaxDD: {m.max_drawdown_pct:.2f}%")
    else:
        print("\nNo outliers > 30% found. The DD must come from another source — "
              "investigate window-by-window equity curves.")


if __name__ == "__main__":
    main()
