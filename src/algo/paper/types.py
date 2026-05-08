"""Paper-trading types: contracts, orders, fills, positions, market snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from algo.backtest.costs import Side
from algo.options.costs import FNOCostBreakdown
from algo.options.greeks import Right


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    strike: int
    expiry: pd.Timestamp  # 15:30 IST on expiry day
    right: Right
    lot_size: int = 75

    def key(self) -> str:
        e = self.expiry.strftime("%Y%m%d")
        return f"{self.underlying}-{e}-{self.strike}-{self.right.value}"


@dataclass(frozen=True)
class MarketSnapshot:
    """Cross-section of market state at a single timestamp.

    The strategy and broker see the world only through these snapshots.
    Real adapters (live or recorded) populate `option_prices` from market data;
    synthetic adapters leave it empty and the broker falls back to BSM.
    """
    timestamp: pd.Timestamp
    spot: float
    iv: float = 0.15
    option_prices: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderId:
    value: int


@dataclass(frozen=True)
class Order:
    id: OrderId
    contract: OptionContract
    side: Side
    lots: int
    submitted_at: pd.Timestamp
    tag: str = ""

    def __post_init__(self) -> None:
        if self.lots <= 0:
            raise ValueError("lots must be positive")


@dataclass(frozen=True)
class Fill:
    order_id: OrderId
    contract: OptionContract
    side: Side
    lots: int
    fill_premium: float           # per-share, after slippage
    fill_time: pd.Timestamp
    quoted_premium: float         # per-share, before slippage
    costs: FNOCostBreakdown
    tag: str = ""

    @property
    def cash_flow(self) -> float:
        """Cash impact of the fill including all costs.

        BUY: cash_flow < 0 (premium paid out + costs)
        SELL: cash_flow > 0 (premium received - costs)
        """
        gross = self.fill_premium * self.contract.lot_size * self.lots
        if self.side is Side.BUY:
            return -(gross + self.costs.total)
        return gross - self.costs.total


@dataclass
class Position:
    contract: OptionContract
    lots: int = 0                 # net signed lots: + long, - short
    avg_entry_premium: float = 0.0
    realised_pnl: float = 0.0

    def is_flat(self) -> bool:
        return self.lots == 0

    def mark_to_market(self, current_premium: float) -> float:
        """Unrealised PnL at current premium."""
        if self.lots == 0:
            return 0.0
        # Long: gain if current > entry; Short: gain if current < entry
        return (current_premium - self.avg_entry_premium) * self.contract.lot_size * self.lots
