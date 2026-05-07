"""Strategy interface.

A strategy is a pure function from a price panel to a position panel:
positions_t ∈ {-1, 0, +1} per symbol per day (or fractional for sized strategies).
The backtest engine consumes positions and applies the cost model — strategies
themselves never compute PnL. Keeps research and accounting separate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyConfig:
    name: str
    universe: tuple[str, ...]
    lookback_days: int
    rebalance: str  # "daily" | "weekly" | "monthly"


class Strategy(ABC):
    config: StrategyConfig

    @abstractmethod
    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a (date × symbol) DataFrame of target weights in [-1, 1]."""
        ...
