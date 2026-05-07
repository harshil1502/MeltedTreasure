"""Backtest engine sanity checks on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.backtest.costs import Product
from algo.backtest.engine import BacktestConfig, run_backtest


def _flat_prices(symbols: list[str], n_days: int = 50, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame({s: np.full(n_days, start_price) for s in symbols}, index=idx)


def _rising_prices(symbols: list[str], n_days: int = 50, daily_drift: float = 0.001) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    growth = (1 + daily_drift) ** np.arange(n_days)
    return pd.DataFrame({s: 100.0 * growth for s in symbols}, index=idx)


class TestEngineBasics:
    def test_no_signals_no_trades(self):
        prices = _flat_prices(["A", "B"])
        weights = pd.DataFrame(index=prices.index[:0], columns=prices.columns, dtype=float)
        result = run_backtest(prices=prices, target_weights=weights)
        assert result.trades.empty
        assert result.equity.iloc[-1] == pytest.approx(10_000.0)

    def test_buy_and_hold_flat_market_loses_only_costs(self):
        prices = _flat_prices(["A"])
        weights = pd.DataFrame({"A": [1.0]}, index=[prices.index[5]])
        result = run_backtest(
            prices=prices,
            target_weights=weights,
            config=BacktestConfig(initial_capital=10_000, product=Product.CNC, slippage_bps=5),
        )
        # Only buy executed; no DP charge until sell. Final equity = capital - buy costs.
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["side"] == "BUY"
        assert result.equity.iloc[-1] < 10_000.0
        assert result.equity.iloc[-1] > 9_900.0  # cost should be modest

    def test_rebalance_then_full_exit(self):
        prices = _flat_prices(["A", "B"], n_days=20)
        weights = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=[prices.index[2], prices.index[10]],
        )
        result = run_backtest(prices=prices, target_weights=weights)
        # Two rebalances → 1 buy, then 1 sell + 1 buy (rotate from A to B)
        assert len(result.trades) == 3
        sides = result.trades["side"].tolist()
        assert sides.count("SELL") == 1
        assert sides.count("BUY") == 2

    def test_rising_market_makes_money_after_costs(self):
        prices = _rising_prices(["A"], n_days=100, daily_drift=0.005)  # ~0.5%/day
        weights = pd.DataFrame({"A": [1.0]}, index=[prices.index[5]])
        result = run_backtest(prices=prices, target_weights=weights)
        assert result.equity.iloc[-1] > 10_000.0


class TestWholeShareConstraint:
    def test_skips_dust_position(self):
        prices = _flat_prices(["A"], start_price=10_000.0)
        # Asking for 10% in a stock priced at ₹10,000 → can afford 0 shares, dust skip
        weights = pd.DataFrame({"A": [0.1]}, index=[prices.index[5]])
        result = run_backtest(prices=prices, target_weights=weights)
        assert result.trades.empty
        assert result.equity.iloc[-1] == pytest.approx(10_000.0)

    def test_buys_whole_shares_only(self):
        prices = _flat_prices(["A"], start_price=333.33)  # awkward price
        weights = pd.DataFrame({"A": [1.0]}, index=[prices.index[5]])
        result = run_backtest(prices=prices, target_weights=weights)
        # 10000 / 333.33 ≈ 30 shares; engine should buy whole shares only
        qty = int(result.trades.iloc[0]["qty"])
        assert qty == 30 or qty == 29  # may be 29 if costs nudge over budget
