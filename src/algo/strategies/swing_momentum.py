"""Swing momentum on Nifty 50.

Hypothesis (H1 from research_plan.md): cross-sectional momentum on Indian
large-caps, filtered by trend, has positive expected value after costs at
1–10 day holding periods.

Signal:
1. Compute 20-day total return for each name in the universe
2. Filter: keep only names trading above 50-day SMA (trend regime)
3. Rank cross-sectionally by 20-day return on the rebalance day
4. Equal-weight the top N names; cash otherwise

Rebalance: weekly (Monday close, or first trading day of the week). Daily
rebalance churns too much for ₹10k after costs.

The strategy emits target weights only. PnL accounting and cost handling
live in the backtest engine — strategies stay pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algo.strategies.base import Strategy, StrategyConfig


@dataclass
class SwingMomentumParams:
    lookback: int = 20
    trend_sma: int = 50
    top_n: int = 2          # ₹10k can typically only support 1-2 positions on large-caps
    rebalance_weekday: int = 0  # 0=Mon, 4=Fri


@dataclass
class SwingMomentum(Strategy):
    params: SwingMomentumParams = field(default_factory=SwingMomentumParams)
    config: StrategyConfig = field(
        default_factory=lambda: StrategyConfig(
            name="swing_momentum_n50",
            universe=(),  # set at instantiation by caller
            lookback_days=50,
            rebalance="weekly",
        )
    )

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.sort_index().ffill()
        returns_lb = prices.pct_change(self.params.lookback)
        sma = prices.rolling(self.params.trend_sma, min_periods=self.params.trend_sma).mean()

        in_trend = prices > sma
        ranked_returns = returns_lb.where(in_trend)

        # Cross-sectional rank: rank=1 is the best return that day
        ranks = ranked_returns.rank(axis=1, ascending=False, method="first")
        selected = (ranks <= self.params.top_n).astype(float)

        # Equal-weight selected names; rows with no selection => all zeros (cash)
        n_selected = selected.sum(axis=1)
        weights = selected.div(n_selected.where(n_selected > 0, np.nan), axis=0).fillna(0.0)

        # Restrict to weekly rebalance days only
        rebalance_mask = weights.index.weekday == self.params.rebalance_weekday
        weights = weights.where(
            pd.Series(rebalance_mask, index=weights.index), other=np.nan
        )
        # Drop pre-warmup rows (where SMA wasn't computable)
        warmup = max(self.params.lookback, self.params.trend_sma)
        weights.iloc[:warmup] = np.nan
        return weights.dropna(how="all")
