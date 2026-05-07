"""Re-run robustness checks with cleaned ETF prices.

Confirms (or refutes) the hypothesis that the failing parameter edges from
notebook 04 were data-quality artifacts, not strategy failures.
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
from algo.data.loaders import adjusted_close_panel, clean_split_artifacts
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams

OUT_DIR = Path("data/cache")
PASS_GATE = 0.7


def aggregate_metrics(prices, signal_fn, cfg, windows, warmup):
    results = run_walk_forward(
        prices=prices, signal_fn=signal_fn, config=cfg,
        windows=windows, warmup_days=warmup,
    )
    if not results:
        return None
    rets = pd.concat([r.result.equity.pct_change().dropna() for r in results]).sort_index()
    eq = (1 + rets).cumprod() * cfg.initial_capital
    return compute_metrics(eq)


def main() -> None:
    print("Loading ETF prices and cleaning split artifacts...")
    symbols = to_yfinance(ETF_ROTATION)
    raw = adjusted_close_panel(symbols, start="2014-01-01", end=date.today().isoformat())
    raw.columns = [c.replace(".NS", "") for c in raw.columns]
    raw = raw.dropna(how="all")
    prices = clean_split_artifacts(raw)
    print()

    windows = make_windows(prices.index, train_years=3, test_years=1, step_years=1)
    cfg = BacktestConfig(
        initial_capital=10_000.0, product=Product.CNC, slippage_bps=10.0,
        min_position_inr=2_000.0,
    )

    # 1. Slippage stress on cleaned data
    print("=== Slippage stress (cleaned data) ===")
    rows = []
    base_strat = EtfRotation(params=EtfRotationParams())
    for bps in (5, 10, 15, 25):
        cfg_b = BacktestConfig(
            initial_capital=10_000.0, product=Product.CNC,
            slippage_bps=float(bps), min_position_inr=2_000.0,
        )
        m = aggregate_metrics(prices, base_strat.signals, cfg_b, windows, warmup=200)
        rows.append({
            "slippage_bps": bps, "cagr_pct": round(m.cagr_pct, 2),
            "sharpe": round(m.sharpe, 3), "max_dd_pct": round(m.max_drawdown_pct, 2),
            "passes_gate": m.sharpe > PASS_GATE,
        })
    slip_df = pd.DataFrame(rows)
    print(slip_df.to_string(index=False))
    slip_pass = bool(slip_df.loc[slip_df["slippage_bps"] == 15, "passes_gate"].iloc[0])

    # 2. Parameter grid on cleaned data
    print("\n=== Parameter robustness (cleaned data) ===")
    grid = list(product([42, 63, 84, 126], [100, 150, 200, 250]))
    rows = []
    for lb, sma in grid:
        strat = EtfRotation(params=EtfRotationParams(momentum_lookback=lb, trend_sma=sma))
        m = aggregate_metrics(prices, strat.signals, cfg, windows, warmup=sma)
        rows.append({
            "momentum_lb": lb, "trend_sma": sma,
            "cagr_pct": round(m.cagr_pct, 2),
            "sharpe": round(m.sharpe, 3),
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "passes_gate": m.sharpe > PASS_GATE,
        })
    grid_df = pd.DataFrame(rows)
    print(grid_df.to_string(index=False))
    sharpe_pivot = grid_df.pivot(index="momentum_lb", columns="trend_sma", values="sharpe")
    print("\nSharpe matrix:")
    print(sharpe_pivot.to_string())
    pass_rate = grid_df["passes_gate"].mean()
    grid_pass = pass_rate >= 2 / 3

    # 3. Final verdict
    print("\n=== Final verdict (cleaned data) ===")
    print(f"  Slippage stress at 15 bps:    {'PASS' if slip_pass else 'FAIL'}")
    print(f"  Parameter robustness ({pass_rate:.0%}):     "
          f"{'PASS' if grid_pass else 'FAIL'} (gate: >= 2/3 of grid)")
    print(f"  -> Phase 2 readiness:          "
          f"{'READY' if (slip_pass and grid_pass) else 'NOT READY'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slip_df.to_parquet(OUT_DIR / "robustness_slippage_clean.parquet")
    grid_df.to_parquet(OUT_DIR / "robustness_grid_clean.parquet")


if __name__ == "__main__":
    main()
