"""Opening Range Breakout on Nifty options.

Strategy:
    1. First 15 minutes after market open (09:15-09:30 IST) defines the range.
    2. After 09:30, watch for a 5-min close beyond the range.
    3. On break-up: buy ATM weekly call. On break-down: buy ATM weekly put.
       Buy at the next bar's open; assume 10 bps slippage.
    4. Stop: opposite end of opening range (at index level).
    5. Target: 1.5x range from breakout level.
    6. Time stop: exit by 15:15 IST regardless.
    7. One trade per day max.

Position sizing: ₹10k can typically buy 1 lot of weekly ATM Nifty option.
If premium × lot_size > available capital, skip the day.

Caveats baked into reality:
- Synthetic pricing via BSM with constant IV (real skew/smile not modeled).
- 5m bars: stops can gap inside a bar; assume worst-case fill at stop level.
- yfinance only gives 60 trading days of 5m data; statistical power is low.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import pandas as pd

from algo.backtest.costs import Side
from algo.options.costs import option_leg_cost
from algo.options.greeks import Right
from algo.options.synthetic_pricing import round_strike_to_nifty, synthetic_option_price

NIFTY_LOT = 75
MARKET_OPEN = time(9, 15)
ORB_END = time(9, 30)
TIME_STOP = time(15, 15)


@dataclass
class ORBParams:
    iv: float = 0.15
    target_multiple: float = 1.5
    slippage_bps: float = 10.0
    risk_free: float = 0.065
    dividend_yield: float = 0.012


@dataclass
class TradeRecord:
    date: pd.Timestamp
    direction: str           # "CALL" or "PUT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    spot_entry: float
    spot_exit: float
    strike: int
    premium_entry: float
    premium_exit: float
    pnl_gross: float
    pnl_net: float
    cost: float


def _next_weekly_expiry(dt: pd.Timestamp) -> pd.Timestamp:
    """Nearest Thursday (Nifty weekly expiry). If today is Thursday, today.

    Note: NSE moved Nifty weekly expiry to Tuesday in 2025; verify current
    expiry day before going live.
    """
    days_ahead = (3 - dt.weekday()) % 7  # 3 = Thursday
    expiry_date = dt + pd.Timedelta(days=days_ahead)
    return expiry_date.normalize() + pd.Timedelta(hours=15, minutes=30)


def _minutes_to_expiry(now: pd.Timestamp, expiry: pd.Timestamp) -> int:
    delta = expiry - now
    return max(int(delta.total_seconds() / 60), 1)


def _price_option(*, spot: float, strike: int, now: pd.Timestamp, expiry: pd.Timestamp,
                  right: Right, params: ORBParams) -> float:
    return synthetic_option_price(
        spot=spot, strike=strike,
        minutes_to_expiry=_minutes_to_expiry(now, expiry),
        iv=params.iv, rate=params.risk_free,
        dividend_yield=params.dividend_yield, right=right,
    )


def _trade_one_day(day_bars: pd.DataFrame, params: ORBParams) -> TradeRecord | None:
    """Run the ORB rules on a single day's 5m bars. Returns one trade or None."""
    bars = day_bars.between_time(MARKET_OPEN, TIME_STOP)
    if bars.empty:
        return None

    orb_window = bars.between_time(MARKET_OPEN, ORB_END)
    if len(orb_window) < 2:
        return None
    range_high = orb_window["High"].max()
    range_low = orb_window["Low"].min()
    range_size = range_high - range_low
    if range_size <= 0:
        return None

    # After ORB window, look for the first bar that closes outside the range
    post = bars[bars.index > orb_window.index[-1]]
    if post.empty:
        return None

    breakout_idx = None
    direction: str | None = None
    for ts, bar in post.iterrows():
        if bar["Close"] > range_high:
            breakout_idx, direction = ts, "CALL"
            break
        if bar["Close"] < range_low:
            breakout_idx, direction = ts, "PUT"
            break
    if breakout_idx is None:
        return None

    # Enter on the NEXT bar's open (avoid look-ahead)
    after_breakout = post[post.index > breakout_idx]
    if after_breakout.empty:
        return None
    entry_ts = after_breakout.index[0]
    spot_entry = float(after_breakout.iloc[0]["Open"])

    expiry = _next_weekly_expiry(entry_ts)
    strike = round_strike_to_nifty(spot_entry)
    right = Right.CALL if direction == "CALL" else Right.PUT
    premium_entry = _price_option(
        spot=spot_entry, strike=strike, now=entry_ts, expiry=expiry,
        right=right, params=params,
    )
    if premium_entry <= 0:
        return None

    notional = premium_entry * NIFTY_LOT
    if notional > 10_000:
        # Can't afford 1 lot at ₹10k; skip.
        return None

    # Stops at index level: opposite end of ORB. Target: 1.5x range from breakout.
    if direction == "CALL":
        index_stop = range_low
        index_target = spot_entry + params.target_multiple * range_size
    else:
        index_stop = range_high
        index_target = spot_entry - params.target_multiple * range_size

    intraday = after_breakout
    exit_ts = None
    spot_exit = None
    exit_reason = None
    for ts, bar in intraday.iterrows():
        if direction == "CALL":
            if bar["Low"] <= index_stop:
                exit_ts, spot_exit, exit_reason = ts, index_stop, "stop"
                break
            if bar["High"] >= index_target:
                exit_ts, spot_exit, exit_reason = ts, index_target, "target"
                break
        else:
            if bar["High"] >= index_stop:
                exit_ts, spot_exit, exit_reason = ts, index_stop, "stop"
                break
            if bar["Low"] <= index_target:
                exit_ts, spot_exit, exit_reason = ts, index_target, "target"
                break
    if exit_ts is None:
        # Time-stop at last bar before 15:15
        last_bar = intraday.iloc[-1]
        exit_ts = intraday.index[-1]
        spot_exit = float(last_bar["Close"])
        exit_reason = "time"

    premium_exit = _price_option(
        spot=spot_exit, strike=strike, now=exit_ts, expiry=expiry,
        right=right, params=params,
    )

    entry_costs = option_leg_cost(
        premium=premium_entry, lot_size=NIFTY_LOT, lots=1,
        side=Side.BUY, slippage_bps=params.slippage_bps,
    )
    exit_costs = option_leg_cost(
        premium=premium_exit, lot_size=NIFTY_LOT, lots=1,
        side=Side.SELL, slippage_bps=params.slippage_bps,
    )
    total_cost = entry_costs.total + exit_costs.total

    pnl_gross = (premium_exit - premium_entry) * NIFTY_LOT
    pnl_net = pnl_gross - total_cost

    return TradeRecord(
        date=entry_ts.normalize(),
        direction=direction,
        entry_time=entry_ts, exit_time=exit_ts, exit_reason=exit_reason,
        spot_entry=spot_entry, spot_exit=spot_exit,
        strike=strike, premium_entry=premium_entry, premium_exit=premium_exit,
        pnl_gross=pnl_gross, pnl_net=pnl_net, cost=total_cost,
    )


def run_orb_backtest(bars: pd.DataFrame, params: ORBParams | None = None) -> list[TradeRecord]:
    """Iterate over each trading day in the bar history and emit trades."""
    p = params or ORBParams()
    trades: list[TradeRecord] = []
    for d, day_bars in bars.groupby(bars.index.date):
        trade = _trade_one_day(day_bars, p)
        if trade is not None:
            trades.append(trade)
    return trades
