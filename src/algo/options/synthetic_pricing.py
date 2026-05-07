"""Synthetic option pricing for backtests without real options data.

Uses Black-Scholes-Merton with a constant or vector-supplied IV. For Nifty,
constant IV ≈ 15% is reasonable on average; for tighter backtests pass an
IV series aligned to the index timestamps (e.g., from India VIX).

Limitations vs real intraday options data:
- Bid-ask spread is approximated by the slippage assumption; real spreads on
  OTM strikes can be 1-3% during illiquid periods.
- Skew is ignored: synthetic ATM prices are accurate, but synthetic OTM
  prices systematically underpredict real market because of skew/smile.
- Pin risk near expiry is not modeled.
- Microstructure (order book, queue position) is absent.

Use this for STRATEGY VALIDATION ONLY. Live PnL with these prices is fiction.
"""
from __future__ import annotations

from algo.options.greeks import BSMInputs, Right, price


def synthetic_option_price(
    *,
    spot: float,
    strike: float,
    minutes_to_expiry: int,
    iv: float = 0.15,
    rate: float = 0.065,
    dividend_yield: float = 0.012,
    right: Right = Right.CALL,
) -> float:
    """Price an option in real time given the current index level.

    minutes_to_expiry must be > 0; for expired options use intrinsic value
    via the Greeks module directly.
    """
    minutes_per_year = 365 * 24 * 60
    t = max(minutes_to_expiry / minutes_per_year, 1e-9)
    return price(
        BSMInputs(
            spot=spot, strike=strike, time_to_expiry=t,
            rate=rate, dividend_yield=dividend_yield,
            iv=iv, right=right,
        )
    )


def round_strike_to_nifty(spot: float, step: int = 50) -> int:
    """Snap to the nearest standard Nifty strike grid (50-point increments)."""
    return int(round(spot / step) * step)
