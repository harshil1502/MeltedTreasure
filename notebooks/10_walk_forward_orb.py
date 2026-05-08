"""Run the ORB strategy through the walk-forward harness on yfinance data.

Caveat heaped on caveat:
- yfinance gives ~58 days of intraday data — far too short for a 9-month
  train window. We use shrunk windows (4-week train / 2-week test) just to
  exercise the harness end-to-end.
- The real walk-forward run is gated on Kite Connect data (12+ months).
- Whatever this notebook prints is a smoke test of the harness, not evidence
  about the strategy's edge.
"""
from __future__ import annotations

import pandas as pd

from algo.data.intraday import load_nifty_intraday
from algo.strategies.options_orb import ORBParams, run_orb_backtest
from algo.validation.walk_forward import GauntletCriteria, run_walk_forward


def orb_strategy_fn(window_bars: pd.DataFrame) -> pd.DataFrame:
    """Adapter: take a slice of Nifty bars, return trades DataFrame."""
    trades = run_orb_backtest(window_bars, ORBParams())
    if not trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "pnl_net"])
    return pd.DataFrame([{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "pnl_net": t.pnl_net,
        "pnl_gross": t.pnl_gross,
        "cost": t.cost,
        "direction": t.direction,
    } for t in trades])


def main() -> None:
    print("Loading Nifty 5m bars...")
    bars = load_nifty_intraday(interval="5m", days=58)
    print(f"  {len(bars)} bars; date range: {bars.index.min()} → {bars.index.max()}")

    print("\nRunning walk-forward (4-week train / 2-week test, shrunk for limited data)...")
    result = run_walk_forward(
        bars, orb_strategy_fn,
        train_period=pd.DateOffset(weeks=4),
        test_period=pd.DateOffset(weeks=2),
        step=pd.DateOffset(weeks=2),
        starting_capital=100_000,
        criteria=GauntletCriteria(),
    )

    print(f"\n{result.windows_total} windows produced")
    print(f"{result.windows_passed}/{result.windows_total} windows passed gauntlet criteria\n")

    for i, v in enumerate(result.verdicts, 1):
        w = v.window
        ins = w.train_metrics
        oos = w.test_metrics
        print(f"--- Window {i}: train {w.train_start.date()} → {w.train_end.date()}, "
              f"test → {w.test_end.date()} ---")
        print(f"  IN-SAMPLE   trades={ins.trade_count:3d} WR={ins.win_rate*100:5.1f}% "
              f"PnL=₹{ins.total_pnl:>9.0f} Sharpe={ins.sharpe:>5.2f} MaxDD={ins.max_drawdown_pct:>6.2f}%")
        print(f"  OUT-OF-SAMPLE trades={oos.trade_count:3d} WR={oos.win_rate*100:5.1f}% "
              f"PnL=₹{oos.total_pnl:>9.0f} Sharpe={oos.sharpe:>5.2f} MaxDD={oos.max_drawdown_pct:>6.2f}%")
        gates = (
            f"sharpe={'PASS' if v.sharpe_pass else 'FAIL'} "
            f"win_rate={'PASS' if v.win_rate_pass else 'FAIL'} "
            f"max_dd={'PASS' if v.max_dd_pass else 'FAIL'} "
            f"expectancy={'PASS' if v.expectancy_pass else 'FAIL'}"
        )
        verdict = "PASSED" if v.overall_pass else "FAILED"
        print(f"  Gates: {gates}  →  Window {verdict}")
        for note in v.notes:
            print(f"    note: {note}")
        print()

    print("=" * 60)
    print("HONEST INTERPRETATION")
    print("=" * 60)
    print("This run uses yfinance daily data on shrunk windows because real")
    print("Kite-quality 12-month options data isn't yet available. Treat the")
    print("results as harness sanity, not strategy evidence.")
    print()
    print("Production-grade walk-forward needs:")
    print("  - 12+ months of Kite Connect intraday options data")
    print("  - 9-month train / 3-month test windows")
    print("  - At least 4 non-overlapping windows for statistical weight")
    print("  - Pass rate ≥ 75% to consider the strategy generalisable")


if __name__ == "__main__":
    main()
