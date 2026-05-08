"""Pre-flight robustness checks before paper trading.

Two experiments:
  1. Slippage stress: re-run walk-forward at 5, 10, 15, 25 bps slippage.
     Pass if aggregate OOS Sharpe stays > 0.7 at 15 bps (a realistic
     pessimistic estimate for retail-size ETF orders).

  2. Parameter robustness: run walk-forward over a small grid of
     (momentum_lookback, trend_sma) values. Pass if at least 2/3 of the
     grid clears Sharpe > 0.7 OOS.

If both pass, the strategy graduates to Phase 2 (paper trading).
"""
from __future__ import annotations

from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig
from algo.backtest.metrics import compute_metrics
from algo.backtest.walk_forward import make_windows, run_walk_forward
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams

OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PASS_GATE = 0.7


def aggregate_sharpe(prices, signal_fn, cfg, windows, warmup):
    results = run_walk_forward(
        prices=prices,
        signal_fn=signal_fn,
        config=cfg,
        windows=windows,
        warmup_days=warmup,
    )
    if not results:
        return None
    rets = pd.concat([r.result.equity.pct_change().dropna() for r in results]).sort_index()
    eq = (1 + rets).cumprod() * cfg.initial_capital
    m = compute_metrics(eq)
    return m


def main() -> None:
    print("Loading ETF prices...")
    symbols = to_yfinance(ETF_ROTATION)
    prices = adjusted_close_panel(symbols, start="2014-01-01", end=date.today().isoformat())
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    prices = prices.dropna(how="all")
    windows = make_windows(prices.index, train_years=3, test_years=1, step_years=1)
    print(f"  {len(prices)} days, {len(windows)} OOS windows")

    # ---------- 1. Slippage stress ----------
    print("\n=== 1. Slippage stress ===")
    slip_rows = []
    base_strat = EtfRotation(params=EtfRotationParams())
    for bps in (5, 10, 15, 25):
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            product=Product.CNC,
            slippage_bps=float(bps),
            min_position_inr=2_000.0,
        )
        m = aggregate_sharpe(prices, base_strat.signals, cfg, windows, warmup=200)
        slip_rows.append({
            "slippage_bps": bps,
            "cagr_pct": m.cagr_pct,
            "sharpe": m.sharpe,
            "max_dd_pct": m.max_drawdown_pct,
            "passes_gate": m.sharpe > PASS_GATE,
        })
    slip_df = pd.DataFrame(slip_rows)
    print(slip_df.to_string(index=False))
    slip_pass = bool(slip_df.loc[slip_df["slippage_bps"] == 15, "passes_gate"].iloc[0])
    print(f"\nSlippage stress: {'PASS' if slip_pass else 'FAIL'} (gate: Sharpe > {PASS_GATE} at 15 bps)")

    # ---------- 2. Parameter robustness ----------
    print("\n=== 2. Parameter robustness ===")
    grid = list(product(
        [42, 63, 84, 126],   # momentum lookback (~2, 3, 4, 6 months)
        [100, 150, 200, 250],  # trend SMA
    ))
    grid_rows = []
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        product=Product.CNC,
        slippage_bps=10.0,  # midpoint slippage for robustness check
        min_position_inr=2_000.0,
    )
    for lb, sma in grid:
        strat = EtfRotation(params=EtfRotationParams(momentum_lookback=lb, trend_sma=sma))
        m = aggregate_sharpe(prices, strat.signals, cfg, windows, warmup=sma)
        grid_rows.append({
            "momentum_lb": lb,
            "trend_sma": sma,
            "cagr_pct": round(m.cagr_pct, 2),
            "sharpe": round(m.sharpe, 3),
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "passes_gate": m.sharpe > PASS_GATE,
        })
    grid_df = pd.DataFrame(grid_rows)
    print(grid_df.to_string(index=False))

    sharpe_pivot = grid_df.pivot(index="momentum_lb", columns="trend_sma", values="sharpe")
    print("\nSharpe matrix (rows=momentum_lb, cols=trend_sma):")
    print(sharpe_pivot.to_string())

    pass_rate = grid_df["passes_gate"].mean()
    grid_pass = pass_rate >= 2 / 3
    print(f"\nParameter robustness: {pass_rate:.0%} of grid clears Sharpe > {PASS_GATE}")
    print(f"  {'PASS' if grid_pass else 'FAIL'} (gate: >= 2/3 of grid passes)")

    print("\n=== Final verdict ===")
    overall = slip_pass and grid_pass
    print(f"  Slippage stress:        {'PASS' if slip_pass else 'FAIL'}")
    print(f"  Parameter robustness:   {'PASS' if grid_pass else 'FAIL'}")
    print(f"  -> Phase 2 readiness:   {'READY' if overall else 'NOT READY'}")

    slip_df.to_parquet(OUT_DIR / "robustness_slippage.parquet")
    grid_df.to_parquet(OUT_DIR / "robustness_parameter_grid.parquet")
    print(f"\nArtifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
