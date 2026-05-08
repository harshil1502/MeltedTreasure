"""Trade-level and daily-level performance metrics for strategy evaluation.

Inputs are always a trades DataFrame with at minimum:
    entry_time : pd.Timestamp
    exit_time  : pd.Timestamp
    pnl_net    : float (after costs)

Optionally:
    pnl_gross  : float (before costs)
    cost       : float
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    trade_count: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    profit_factor: float
    largest_trade_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    days_traded: int

    def as_dict(self) -> dict:
        return asdict(self)


def _daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    s = trades.copy()
    s["date"] = pd.to_datetime(s["exit_time"]).dt.normalize()
    return s.groupby("date")["pnl_net"].sum()


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Returns (max_drawdown_inr, max_drawdown_pct_of_peak)."""
    if equity.empty:
        return 0.0, 0.0
    running_max = equity.cummax()
    drawdown = equity - running_max
    dd_pct = drawdown / running_max
    return float(drawdown.min()), float(dd_pct.min())


def _annualised_sharpe(daily: pd.Series, capital: float) -> float:
    if daily.empty or capital <= 0:
        return 0.0
    returns = daily / capital
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def _annualised_sortino(daily: pd.Series, capital: float) -> float:
    if daily.empty or capital <= 0:
        return 0.0
    returns = daily / capital
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    sd_down = downside.std()
    if sd_down == 0 or np.isnan(sd_down):
        return 0.0
    return float(returns.mean() / sd_down * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_metrics(trades: pd.DataFrame, *, starting_capital: float = 100_000) -> Metrics:
    """Compute all metrics from a trades DataFrame."""
    if trades.empty:
        return Metrics(
            trade_count=0, win_rate=0, total_pnl=0,
            avg_win=0, avg_loss=0, win_loss_ratio=0,
            expectancy=0, profit_factor=0, largest_trade_pct=0,
            max_drawdown=0, max_drawdown_pct=0,
            sharpe=0, sortino=0, days_traded=0,
        )

    pnl = trades["pnl_net"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total = float(pnl.sum())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else float("inf") if avg_win > 0 else 0.0

    # Expectancy: avg PnL per trade
    expectancy = float(pnl.mean())

    # Profit factor: gross wins / |gross losses|
    gross_wins = float(wins.sum()) if len(wins) else 0.0
    gross_losses = float(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_wins / abs(gross_losses) if gross_losses < 0 else (
        float("inf") if gross_wins > 0 else 0.0
    )

    # Largest single trade as % of total PnL
    largest_pct = (float(pnl.abs().max()) / abs(total) * 100) if total != 0 else 0.0

    # Daily aggregates for risk metrics
    daily = _daily_pnl(trades)
    equity = starting_capital + daily.cumsum()
    dd_inr, dd_pct = _max_drawdown(equity)
    sharpe = _annualised_sharpe(daily, starting_capital)
    sortino = _annualised_sortino(daily, starting_capital)

    return Metrics(
        trade_count=len(trades),
        win_rate=len(wins) / len(trades),
        total_pnl=total,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=wl_ratio,
        expectancy=expectancy,
        profit_factor=profit_factor,
        largest_trade_pct=largest_pct,
        max_drawdown=dd_inr,
        max_drawdown_pct=dd_pct * 100,
        sharpe=sharpe,
        sortino=sortino,
        days_traded=len(daily),
    )
