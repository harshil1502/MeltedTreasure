"""Walk-forward validation utilities.

Walk-forward = sequence of (train_window, test_window) pairs that roll forward
in time. The strategy runs on each test window using only data available up to
the start of that window; OOS results are concatenated.

For the ETF rotation strategy specifically there is no fittable parameter set —
the strategy uses fixed lookbacks. Walk-forward here is therefore an
out-of-sample regime-robustness check rather than a hyperparameter search:
each test window starts the backtest with a fresh ₹10k and we compare results
across windows. If Sharpe collapses in a window, we learn the strategy's
edge is regime-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from algo.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from algo.backtest.metrics import Metrics, compute_metrics


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(
    index: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[WalkForwardWindow]:
    """Generate rolling walk-forward windows over `index`.

    Each window has `train_years` of training data immediately followed by
    `test_years` of testing data. Windows step forward by `step_years`.
    """
    if len(index) == 0:
        return []
    start = index[0]
    end = index[-1]
    windows: list[WalkForwardWindow] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(years=step_years)
    return windows


@dataclass
class WindowResult:
    window: WalkForwardWindow
    result: BacktestResult
    metrics: Metrics


def run_walk_forward(
    *,
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    config: BacktestConfig,
    windows: list[WalkForwardWindow],
    warmup_days: int = 200,
) -> list[WindowResult]:
    """Run the strategy on each test window with a fresh capital base.

    Signals are generated using the full price history available up to and
    including the test window — the strategy still gets its lookback context
    from the training data, so signals on test_start are well-defined. The
    backtest itself only consumes prices/weights inside [test_start, test_end].
    """
    results: list[WindowResult] = []
    for w in windows:
        # Provide enough warmup before test_start for signal computation
        warmup_start = w.test_start - pd.Timedelta(days=int(warmup_days * 1.6))
        ctx_prices = prices.loc[warmup_start : w.test_end]
        all_signals = signal_fn(ctx_prices)
        # Restrict signals AND prices to the test window
        oos_prices = ctx_prices.loc[w.test_start : w.test_end]
        oos_weights = all_signals.loc[
            (all_signals.index >= w.test_start) & (all_signals.index <= w.test_end)
        ]
        if oos_weights.empty or oos_prices.empty:
            continue
        bt = run_backtest(prices=oos_prices, target_weights=oos_weights, config=config)
        m = compute_metrics(bt.equity, bt.trades)
        results.append(WindowResult(window=w, result=bt, metrics=m))
    return results


def summarize_windows(results: list[WindowResult]) -> pd.DataFrame:
    rows = []
    for wr in results:
        rows.append(
            {
                "test_start": wr.window.test_start.date(),
                "test_end": wr.window.test_end.date(),
                "cagr_pct": wr.metrics.cagr_pct,
                "sharpe": wr.metrics.sharpe,
                "max_dd_pct": wr.metrics.max_drawdown_pct,
                "n_trades": wr.metrics.n_trades,
                "total_cost": wr.result.cost_total,
                "final_equity": float(wr.result.equity.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)
