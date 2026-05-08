"""Walk-forward harness tests using synthetic data + deterministic strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algo.validation.walk_forward import (
    GauntletCriteria,
    evaluate_window,
    run_walk_forward,
)


def make_data(months: int = 24, freq: str = "D") -> pd.DataFrame:
    """24 months of synthetic daily index bars."""
    idx = pd.bdate_range(start="2024-01-01", periods=months * 21, freq=freq)
    rng = np.random.default_rng(42)
    closes = 24000 + rng.normal(0, 50, len(idx)).cumsum()
    return pd.DataFrame({"Close": closes, "Open": closes, "High": closes + 30,
                         "Low": closes - 30}, index=idx)


def constant_winner_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """Trades every 5th day, always wins ₹500. Should pass all gates."""
    days = data.index[::5]
    return pd.DataFrame({
        "entry_time": days, "exit_time": days, "pnl_net": [500] * len(days),
    })


def overfit_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """Returns wins only when the data starts on a Monday — represents an
    accidentally overfit signal that doesn't generalise. Passes train if
    it starts Monday, fails test if it doesn't, and vice versa.
    """
    if data.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "pnl_net"])
    starts_monday = data.index[0].weekday() == 0
    n = max(len(data) // 5, 1)
    days = data.index[::5][:n]
    pnl = 500 if starts_monday else -300
    return pd.DataFrame({
        "entry_time": days, "exit_time": days, "pnl_net": [pnl] * len(days),
    })


def collapsing_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """Wins on the first half of any window, loses on the second half.
    Train sees mostly wins (start), test sees mostly losses (end). Designed
    to fail the OOS expectancy gate.
    """
    if data.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "pnl_net"])
    days = data.index[::5]
    midpoint = len(days) // 2
    pnls = [500] * midpoint + [-700] * (len(days) - midpoint)
    return pd.DataFrame({
        "entry_time": days, "exit_time": days, "pnl_net": pnls,
    })


class TestRunWalkForward:
    def test_produces_windows(self):
        data = make_data(months=24)
        result = run_walk_forward(
            data, constant_winner_strategy,
            train_period=pd.DateOffset(months=9),
            test_period=pd.DateOffset(months=3),
        )
        # 24 business months ≈ 12 calendar months, so window count is sensitive
        # to exact step alignment. At minimum we should get 2 non-overlapping
        # windows.
        assert result.windows_total >= 2

    def test_constant_winner_passes_most_windows(self):
        data = make_data(months=24)
        result = run_walk_forward(
            data, constant_winner_strategy,
            train_period=pd.DateOffset(months=9),
            test_period=pd.DateOffset(months=3),
        )
        # Constant winner has 0 variance → Sharpe == 0 by our convention,
        # so the strict gauntlet won't pass. Check expectancy gate.
        assert all(v.expectancy_pass for v in result.verdicts)

    def test_collapsing_strategy_fails(self):
        """A strategy that wins early and loses late within each window
        should fail the OOS expectancy gate (test slice sees only losses).
        """
        data = make_data(months=24)
        result = run_walk_forward(
            data, collapsing_strategy,
            train_period=pd.DateOffset(months=9),
            test_period=pd.DateOffset(months=3),
        )
        assert result.windows_passed < result.windows_total

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            run_walk_forward(
                pd.DataFrame(),
                constant_winner_strategy,
                train_period=pd.DateOffset(months=9),
                test_period=pd.DateOffset(months=3),
            )


class TestEvaluateWindow:
    def test_sharpe_pass_when_oos_meets_ratio(self):
        from algo.validation.metrics import Metrics
        from algo.validation.walk_forward import WindowResult

        ins = Metrics(
            trade_count=100, win_rate=0.6, total_pnl=10000,
            avg_win=200, avg_loss=-100, win_loss_ratio=2,
            expectancy=100, profit_factor=2, largest_trade_pct=10,
            max_drawdown=-1000, max_drawdown_pct=-2,
            sharpe=2.0, sortino=2.5, days_traded=60,
        )
        oos_pass = Metrics(  # 75% of IS sharpe
            trade_count=30, win_rate=0.55, total_pnl=2500,
            avg_win=180, avg_loss=-110, win_loss_ratio=1.6,
            expectancy=80, profit_factor=1.8, largest_trade_pct=12,
            max_drawdown=-1200, max_drawdown_pct=-2.4,
            sharpe=1.5, sortino=1.8, days_traded=20,
        )
        w = WindowResult(
            train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-09-30"),
            test_start=pd.Timestamp("2024-10-01"), test_end=pd.Timestamp("2024-12-31"),
            train_metrics=ins, test_metrics=oos_pass,
        )
        v = evaluate_window(w, GauntletCriteria())
        assert v.sharpe_pass
        assert v.win_rate_pass
        assert v.max_dd_pass
        assert v.expectancy_pass
        assert v.overall_pass

    def test_sharpe_fail_when_oos_drops_too_much(self):
        from algo.validation.metrics import Metrics
        from algo.validation.walk_forward import WindowResult

        ins = Metrics(
            trade_count=100, win_rate=0.6, total_pnl=10000,
            avg_win=200, avg_loss=-100, win_loss_ratio=2, expectancy=100,
            profit_factor=2, largest_trade_pct=10,
            max_drawdown=-1000, max_drawdown_pct=-2,
            sharpe=2.0, sortino=2.5, days_traded=60,
        )
        oos_fail = Metrics(  # only 30% of IS sharpe
            trade_count=30, win_rate=0.55, total_pnl=500,
            avg_win=180, avg_loss=-110, win_loss_ratio=1.6, expectancy=20,
            profit_factor=1.1, largest_trade_pct=15,
            max_drawdown=-1500, max_drawdown_pct=-3,
            sharpe=0.6, sortino=0.7, days_traded=20,
        )
        w = WindowResult(
            train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-09-30"),
            test_start=pd.Timestamp("2024-10-01"), test_end=pd.Timestamp("2024-12-31"),
            train_metrics=ins, test_metrics=oos_fail,
        )
        v = evaluate_window(w, GauntletCriteria())
        assert not v.sharpe_pass
        assert not v.overall_pass

    def test_max_dd_fail_when_oos_too_deep(self):
        from algo.validation.metrics import Metrics
        from algo.validation.walk_forward import WindowResult

        ins = Metrics(
            trade_count=100, win_rate=0.6, total_pnl=10000,
            avg_win=200, avg_loss=-100, win_loss_ratio=2, expectancy=100,
            profit_factor=2, largest_trade_pct=10,
            max_drawdown=-1000, max_drawdown_pct=-2,
            sharpe=2.0, sortino=2.5, days_traded=60,
        )
        # OOS DD 4% vs IS 2% → 2x → fails 1.5x criterion
        oos = Metrics(
            trade_count=30, win_rate=0.55, total_pnl=2000,
            avg_win=180, avg_loss=-110, win_loss_ratio=1.6, expectancy=80,
            profit_factor=1.8, largest_trade_pct=12,
            max_drawdown=-2000, max_drawdown_pct=-4.0,
            sharpe=1.5, sortino=1.8, days_traded=20,
        )
        w = WindowResult(
            train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-09-30"),
            test_start=pd.Timestamp("2024-10-01"), test_end=pd.Timestamp("2024-12-31"),
            train_metrics=ins, test_metrics=oos,
        )
        v = evaluate_window(w, GauntletCriteria())
        assert not v.max_dd_pass
        assert not v.overall_pass
