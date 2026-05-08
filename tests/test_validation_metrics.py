"""Tests for performance metrics."""
from __future__ import annotations

import pandas as pd
import pytest

from algo.validation.metrics import compute_metrics


def make_trades(pnls: list[float], start_date: str = "2026-01-05") -> pd.DataFrame:
    """Build a trades DataFrame with one trade per business day."""
    n = len(pnls)
    days = pd.bdate_range(start=start_date, periods=n, freq="B")
    return pd.DataFrame({
        "entry_time": days,
        "exit_time": days + pd.Timedelta(hours=4),
        "pnl_net": pnls,
    })


class TestEmptyAndBasic:
    def test_empty_trades_returns_zeros(self):
        m = compute_metrics(pd.DataFrame())
        assert m.trade_count == 0
        assert m.total_pnl == 0
        assert m.sharpe == 0

    def test_single_winning_trade(self):
        m = compute_metrics(make_trades([1000]))
        assert m.trade_count == 1
        assert m.win_rate == 1.0
        assert m.total_pnl == 1000

    def test_all_losses_negative_expectancy(self):
        m = compute_metrics(make_trades([-100, -200, -150]))
        assert m.win_rate == 0
        assert m.expectancy < 0
        assert m.profit_factor == 0


class TestWinLossMetrics:
    def test_win_rate_50pct(self):
        m = compute_metrics(make_trades([100, -100, 200, -150]))
        assert m.win_rate == 0.5

    def test_win_loss_ratio(self):
        m = compute_metrics(make_trades([300, -100, 300, -100]))
        # avg_win=300, avg_loss=-100 → ratio 3.0
        assert m.win_loss_ratio == pytest.approx(3.0)

    def test_profit_factor(self):
        m = compute_metrics(make_trades([200, 200, -100]))
        # gross wins 400, gross losses 100 → PF 4.0
        assert m.profit_factor == pytest.approx(4.0)


class TestRiskMetrics:
    def test_max_drawdown_negative(self):
        m = compute_metrics(make_trades([1000, -800, -600, 500]))
        # equity peaks at 100k+1000=101000, drops to 101000-800-600=99600
        # so DD = -1400 from peak
        assert m.max_drawdown < 0
        assert m.max_drawdown == pytest.approx(-1400, abs=1)

    def test_no_drawdown_for_monotone_wins(self):
        m = compute_metrics(make_trades([100, 200, 300, 400]))
        assert m.max_drawdown == 0

    def test_sharpe_positive_for_consistent_winner(self):
        m = compute_metrics(make_trades([500] * 50))
        # Constant positive returns -> std=0 -> sharpe undefined but we return 0
        assert m.sharpe == 0  # by our convention when std is zero

    def test_sharpe_positive_for_noisy_positive_winner(self):
        # Mostly small wins with a few losses
        pnls = [200, 100, -50, 150, 200, -100, 250, 150, 200, 100] * 5
        m = compute_metrics(make_trades(pnls))
        assert m.sharpe > 0


class TestLargestTradeMetric:
    def test_largest_trade_pct_when_one_dominates(self):
        # 10000 winner dominates total PnL (similar to ORB's Mar 19 trade)
        m = compute_metrics(make_trades([10000, 100, -200, 100, -100]))
        # total = 9900, biggest abs = 10000 -> 101%
        assert m.largest_trade_pct > 100
