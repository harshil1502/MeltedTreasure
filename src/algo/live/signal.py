"""Generate the current target allocation from the ETF rotation strategy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from algo.data.loaders import adjusted_close_panel, clean_split_artifacts
from algo.data.universe import ETF_ROTATION, to_yfinance
from algo.strategies.etf_rotation import EtfRotation, EtfRotationParams


@dataclass
class CurrentSignal:
    as_of: date
    is_rebalance_day: bool
    target_weights: dict[str, float]
    last_rebalance_date: date | None
    underlying_prices: dict[str, float]


def generate_current_signal(
    *,
    history_start: str = "2014-01-01",
    params: EtfRotationParams | None = None,
) -> CurrentSignal:
    """Pull fresh ETF prices, clean, run the strategy, return current target.

    Returns the most-recent rebalance signal regardless of whether today is a
    rebalance day. The caller decides whether to act on it (i.e. on the last
    trading day of the month, place trades to hit the target weights).
    """
    params = params or EtfRotationParams()
    symbols = to_yfinance(ETF_ROTATION)
    raw = adjusted_close_panel(symbols, start=history_start, end=date.today().isoformat())
    raw.columns = [c.replace(".NS", "") for c in raw.columns]
    raw = raw.dropna(how="all")
    prices = clean_split_artifacts(raw)

    strat = EtfRotation(params=params)
    weights_df = strat.signals(prices)
    if weights_df.empty:
        raise RuntimeError("strategy produced no signals (insufficient history)")

    last_rebal = weights_df.index[-1]
    last_weights = weights_df.iloc[-1].to_dict()
    last_weights = {k: round(v, 6) for k, v in last_weights.items() if v > 0}

    today_idx = prices.index[-1]
    return CurrentSignal(
        as_of=today_idx.date(),
        is_rebalance_day=last_rebal.date() == today_idx.date(),
        target_weights=last_weights,
        last_rebalance_date=last_rebal.date(),
        underlying_prices={s: float(prices[s].iloc[-1]) for s in prices.columns},
    )
