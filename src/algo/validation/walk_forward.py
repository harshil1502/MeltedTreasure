"""Walk-forward validation harness — Stage 2 of VALIDATION_GAUNTLET.md.

Splits a long time series into rolling (train, test) windows. Runs a strategy
on each window and computes metrics for both train (in-sample) and test
(out-of-sample) periods. Compares them via the gauntlet criteria:

    OOS Sharpe ≥ 70% of in-sample Sharpe
    OOS win rate within 8pp of in-sample
    OOS max drawdown ≤ 1.5x in-sample max drawdown

If a strategy passes those criteria across most windows, it likely generalises.
If it fails, the in-sample edge was fitted to noise.

This harness is intentionally simple: it runs the SAME strategy_fn on train
and test windows. For parameter-tuned strategies, the user supplies a
strategy_fn that internally tunes on the train slice it receives. A future
extension can split that into separate `tune` and `apply` callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from algo.validation.metrics import Metrics, compute_metrics

StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Metrics
    test_metrics: Metrics

    def degradation_summary(self) -> dict:
        """How much did out-of-sample performance fall vs in-sample?"""
        ins = self.train_metrics
        oos = self.test_metrics
        def safe_ratio(a: float, b: float) -> float:
            if b == 0:
                return float("inf") if a == 0 else 0.0
            return a / b
        return {
            "sharpe_ratio": safe_ratio(oos.sharpe, ins.sharpe) if ins.sharpe > 0 else None,
            "win_rate_delta_pp": (oos.win_rate - ins.win_rate) * 100,
            "max_dd_pct_ratio": safe_ratio(oos.max_drawdown_pct, ins.max_drawdown_pct)
                                if ins.max_drawdown_pct < 0 else None,
            "expectancy_delta": oos.expectancy - ins.expectancy,
        }


@dataclass
class GauntletCriteria:
    oos_sharpe_min_ratio: float = 0.70   # OOS Sharpe must be ≥ 70% of IS
    oos_win_rate_max_drop_pp: float = 8  # OOS win rate within 8pp below IS
    oos_max_dd_ratio: float = 1.5        # OOS max DD ≤ 1.5x IS max DD
    require_oos_positive_expectancy: bool = True


@dataclass
class WindowVerdict:
    window: WindowResult
    sharpe_pass: bool
    win_rate_pass: bool
    max_dd_pass: bool
    expectancy_pass: bool
    overall_pass: bool
    notes: list[str] = field(default_factory=list)


def evaluate_window(window: WindowResult, criteria: GauntletCriteria) -> WindowVerdict:
    notes: list[str] = []
    ins, oos = window.train_metrics, window.test_metrics

    # Sharpe ratio gate
    if ins.sharpe <= 0:
        sharpe_pass = oos.sharpe >= 0
        notes.append("in-sample Sharpe ≤ 0; gate softened to OOS Sharpe ≥ 0")
    else:
        sharpe_pass = oos.sharpe >= criteria.oos_sharpe_min_ratio * ins.sharpe

    # Win rate gate
    win_rate_pass = oos.win_rate >= ins.win_rate - criteria.oos_win_rate_max_drop_pp / 100

    # Max DD gate
    if ins.max_drawdown_pct >= 0:
        max_dd_pass = True
        notes.append("in-sample had no drawdown; gate trivially passed")
    else:
        max_dd_pass = oos.max_drawdown_pct >= ins.max_drawdown_pct * criteria.oos_max_dd_ratio

    # Expectancy gate
    if criteria.require_oos_positive_expectancy:
        expectancy_pass = oos.expectancy > 0
    else:
        expectancy_pass = True

    overall = sharpe_pass and win_rate_pass and max_dd_pass and expectancy_pass

    return WindowVerdict(
        window=window,
        sharpe_pass=sharpe_pass, win_rate_pass=win_rate_pass,
        max_dd_pass=max_dd_pass, expectancy_pass=expectancy_pass,
        overall_pass=overall, notes=notes,
    )


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    verdicts: list[WindowVerdict] = field(default_factory=list)

    @property
    def windows_passed(self) -> int:
        return sum(1 for v in self.verdicts if v.overall_pass)

    @property
    def windows_total(self) -> int:
        return len(self.verdicts)

    def summary(self) -> dict:
        return {
            "windows_total": self.windows_total,
            "windows_passed": self.windows_passed,
            "pass_rate": self.windows_passed / self.windows_total if self.windows_total else 0,
        }


def run_walk_forward(
    data: pd.DataFrame,
    strategy_fn: StrategyFn,
    *,
    train_period: pd.DateOffset,
    test_period: pd.DateOffset,
    step: Optional[pd.DateOffset] = None,
    starting_capital: float = 100_000,
    criteria: Optional[GauntletCriteria] = None,
) -> WalkForwardResult:
    """Run rolling walk-forward.

    Args:
        data: time-indexed DataFrame (any granularity; bars or ticks)
        strategy_fn: takes a slice of data, returns a trades DataFrame with
                     at least entry_time, exit_time, pnl_net columns
        train_period: e.g. pd.DateOffset(months=9)
        test_period:  e.g. pd.DateOffset(months=3)
        step:         how far to advance window-start each iteration.
                      Default = test_period (non-overlapping test windows).
        starting_capital: used for Sharpe/drawdown
        criteria: gauntlet thresholds; defaults to project standard
    """
    if data.empty:
        raise ValueError("data is empty")
    step = step or test_period
    criteria = criteria or GauntletCriteria()

    data = data.sort_index()
    start = data.index.min()
    end = data.index.max()
    if hasattr(start, "tz_convert") and start.tz is not None:
        # Ensure naive comparisons go through pandas, which handles tz
        pass

    windows: list[WindowResult] = []
    cursor = pd.Timestamp(start)
    while True:
        train_end = cursor + train_period
        test_end = train_end + test_period
        if test_end > end:
            break

        train_slice = data.loc[cursor:train_end]
        test_slice = data.loc[train_end:test_end]
        if train_slice.empty or test_slice.empty:
            cursor = cursor + step
            continue

        train_trades = strategy_fn(train_slice)
        test_trades = strategy_fn(test_slice)

        train_metrics = compute_metrics(train_trades, starting_capital=starting_capital)
        test_metrics = compute_metrics(test_trades, starting_capital=starting_capital)

        windows.append(WindowResult(
            train_start=pd.Timestamp(cursor), train_end=pd.Timestamp(train_end),
            test_start=pd.Timestamp(train_end), test_end=pd.Timestamp(test_end),
            train_metrics=train_metrics, test_metrics=test_metrics,
        ))
        cursor = cursor + step

    verdicts = [evaluate_window(w, criteria) for w in windows]
    return WalkForwardResult(windows=windows, verdicts=verdicts)
