"""ETF rotation across asset classes.

Hypothesis (H2 from research_plan.md): rotating across asset-class ETFs
(equity broad-market, equity mid-cap, gold, liquid debt) by 3-month momentum
captures regime shifts and reduces drawdowns versus a single-asset hold.

Universe:
- NIFTYBEES   — Nifty 50 equity
- JUNIORBEES  — Nifty Next 50 equity
- GOLDBEES    — Gold
- LIQUIDBEES  — Liquid debt proxy (acts as cash sleeve in risk-off regimes)

Signal:
1. On the last trading day of each calendar month, rank the four by 63-day
   (~3 month) total return.
2. Hold the top-1 ETF for the next month.
3. Risk-off filter: if the chosen ETF is below its 200-day SMA AND is not
   LIQUIDBEES, force allocation to LIQUIDBEES instead. Prevents holding
   crashing equity in bear markets.

Why this should work at ₹10k:
- Monthly rebalance: ~12 buys + ~12 sells/year, max
- Holding only one ETF at a time: DP charge hits at most once per rebalance
- Zero ETF delivery brokerage on Zerodha
- 200-SMA filter is the simplest defensible regime-shift mechanism
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algo.strategies.base import Strategy, StrategyConfig


@dataclass
class EtfRotationParams:
    momentum_lookback: int = 63
    trend_sma: int = 200
    risk_off_asset: str = "LIQUIDBEES"
    top_n: int = 1


@dataclass
class EtfRotation(Strategy):
    params: EtfRotationParams = field(default_factory=EtfRotationParams)
    config: StrategyConfig = field(
        default_factory=lambda: StrategyConfig(
            name="etf_rotation_v1",
            universe=("NIFTYBEES", "JUNIORBEES", "GOLDBEES", "LIQUIDBEES"),
            lookback_days=200,
            rebalance="monthly",
        )
    )

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.sort_index().ffill()
        momentum = prices.pct_change(self.params.momentum_lookback)
        sma = prices.rolling(self.params.trend_sma, min_periods=self.params.trend_sma).mean()
        above_trend = prices > sma

        # Pick last trading day of each calendar month from the price index
        month_end_mask = prices.index.to_series().groupby(
            prices.index.to_period("M")
        ).transform("max") == prices.index
        rebal_dates = prices.index[month_end_mask]

        weights = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
        risk_off = self.params.risk_off_asset

        for d in rebal_dates:
            if d not in momentum.index:
                continue
            mom_row = momentum.loc[d].dropna()
            trend_row = above_trend.loc[d] if d in above_trend.index else None
            if mom_row.empty:
                continue
            ranked = mom_row.sort_values(ascending=False)
            picks = ranked.head(self.params.top_n).index.tolist()

            # Apply risk-off filter: if any pick is below trend, swap to LIQUIDBEES
            final_picks = []
            for p in picks:
                if (
                    p != risk_off
                    and trend_row is not None
                    and p in trend_row.index
                    and not bool(trend_row[p])
                ):
                    final_picks.append(risk_off)
                else:
                    final_picks.append(p)

            for p in final_picks:
                weights.loc[d, p] += 1.0 / len(final_picks)

        # Drop pre-warmup rows
        warmup = max(self.params.momentum_lookback, self.params.trend_sma)
        weights = weights.loc[weights.index >= prices.index[warmup]]
        return weights.replace(0.0, np.nan).dropna(how="all").fillna(0.0)
