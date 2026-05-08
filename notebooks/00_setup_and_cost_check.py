"""Phase 0 sanity-check script.

Runs as a script so it works without Jupyter. To convert to a notebook:
    jupytext --to ipynb notebooks/00_setup_and_cost_check.py

Validates two things:
1. yfinance can fetch a Nifty 50 daily series end-to-end
2. The cost model produces sane numbers at ₹10k account size
"""
from __future__ import annotations

from datetime import date

from algo.backtest.costs import Product, cost_drag_pct
from algo.data.loaders import adjusted_close_panel
from algo.data.universe import NIFTY_50, to_yfinance
from algo.risk.sizing import fixed_fractional


def check_data() -> None:
    symbols = to_yfinance(NIFTY_50[:3])  # RELIANCE, TCS-ish slice for a quick fetch
    print(f"Fetching last 30 days of {symbols} from yfinance...")
    panel = adjusted_close_panel(symbols, start="2024-12-01", end=date.today().isoformat())
    print(panel.tail())
    assert not panel.empty, "yfinance returned empty"


def check_costs() -> None:
    capital = 10_000.0
    print(f"\nCost-drag table for ₹{capital:,.0f} account:")
    print(f"{'price':>8} {'qty':>5} {'notional':>10} {'MIS drag %':>12} {'CNC drag %':>12}")
    for price in (50, 100, 250, 500, 1000, 2000):
        sized = fixed_fractional(capital=capital, price=price, fraction=1.0)
        if sized.skipped_reason:
            print(f"{price:>8} skipped: {sized.skipped_reason}")
            continue
        mis = cost_drag_pct(
            entry_price=price, exit_price=price, quantity=sized.quantity,
            product=Product.MIS, slippage_bps=5,
        )
        cnc = cost_drag_pct(
            entry_price=price, exit_price=price, quantity=sized.quantity,
            product=Product.CNC, slippage_bps=5,
        )
        print(f"{price:>8} {sized.quantity:>5} {sized.notional:>10.0f} {mis:>12.3f} {cnc:>12.3f}")


if __name__ == "__main__":
    check_data()
    check_costs()
