"""Black-Scholes-Merton pricing, Greeks, and implied volatility solver.

Indian-market specifics:
- Nifty/BankNifty options are European-style (cash-settled at expiry); BSM
  is the right model.
- Underlying: Nifty 50 spot index. Use spot for pricing intraday/short-dated
  options. For longer-dated, use futures basis-adjusted spot.
- Risk-free rate: 91-day T-bill yield ~6.5% nominal (FY25). Time-of-day matters
  less than IV for retail use; pass `r` as an annual fraction.
- Dividend yield on Nifty: ~1.2% (FY25). For European index options the
  Black-Scholes-Merton formula with continuous yield is appropriate.
- Day-count: actual/365, expressed as fraction of a year (T = days_to_expiry/365).
  For sub-day precision, T = (seconds_to_expiry / (365*24*3600)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from scipy.stats import norm


class Right(str, Enum):
    CALL = "CE"
    PUT = "PE"


@dataclass(frozen=True)
class BSMInputs:
    spot: float
    strike: float
    time_to_expiry: float  # years; e.g. 0.0192 ≈ 7 days
    rate: float            # annual continuous, e.g. 0.065
    dividend_yield: float  # annual continuous, e.g. 0.012
    iv: float              # annual, e.g. 0.15 = 15%
    right: Right


def _d1(i: BSMInputs) -> float:
    return (
        math.log(i.spot / i.strike)
        + (i.rate - i.dividend_yield + 0.5 * i.iv * i.iv) * i.time_to_expiry
    ) / (i.iv * math.sqrt(i.time_to_expiry))


def _d2(i: BSMInputs) -> float:
    return _d1(i) - i.iv * math.sqrt(i.time_to_expiry)


def price(i: BSMInputs) -> float:
    """Black-Scholes-Merton price of a European option."""
    if i.time_to_expiry <= 0 or i.iv <= 0:
        # At expiry: intrinsic value
        if i.right is Right.CALL:
            return max(0.0, i.spot - i.strike)
        return max(0.0, i.strike - i.spot)
    d1 = _d1(i)
    d2 = _d2(i)
    if i.right is Right.CALL:
        return (
            i.spot * math.exp(-i.dividend_yield * i.time_to_expiry) * norm.cdf(d1)
            - i.strike * math.exp(-i.rate * i.time_to_expiry) * norm.cdf(d2)
        )
    return (
        i.strike * math.exp(-i.rate * i.time_to_expiry) * norm.cdf(-d2)
        - i.spot * math.exp(-i.dividend_yield * i.time_to_expiry) * norm.cdf(-d1)
    )


def delta(i: BSMInputs) -> float:
    if i.time_to_expiry <= 0:
        if i.right is Right.CALL:
            return 1.0 if i.spot > i.strike else 0.0
        return -1.0 if i.spot < i.strike else 0.0
    d1 = _d1(i)
    factor = math.exp(-i.dividend_yield * i.time_to_expiry)
    return factor * norm.cdf(d1) if i.right is Right.CALL else factor * (norm.cdf(d1) - 1)


def gamma(i: BSMInputs) -> float:
    if i.time_to_expiry <= 0 or i.iv <= 0:
        return 0.0
    d1 = _d1(i)
    return (
        math.exp(-i.dividend_yield * i.time_to_expiry)
        * norm.pdf(d1)
        / (i.spot * i.iv * math.sqrt(i.time_to_expiry))
    )


def vega(i: BSMInputs) -> float:
    """Vega per 1.0 change in IV (i.e. per 100 vol points). Divide by 100 for per-vol-point."""
    if i.time_to_expiry <= 0:
        return 0.0
    d1 = _d1(i)
    return (
        i.spot
        * math.exp(-i.dividend_yield * i.time_to_expiry)
        * norm.pdf(d1)
        * math.sqrt(i.time_to_expiry)
    )


def theta(i: BSMInputs) -> float:
    """Theta per year. Divide by 365 for per-day theta."""
    if i.time_to_expiry <= 0:
        return 0.0
    d1 = _d1(i)
    d2 = _d2(i)
    sqrtt = math.sqrt(i.time_to_expiry)
    common = -i.spot * math.exp(-i.dividend_yield * i.time_to_expiry) * norm.pdf(d1) * i.iv / (
        2 * sqrtt
    )
    if i.right is Right.CALL:
        return (
            common
            - i.rate * i.strike * math.exp(-i.rate * i.time_to_expiry) * norm.cdf(d2)
            + i.dividend_yield * i.spot * math.exp(-i.dividend_yield * i.time_to_expiry) * norm.cdf(d1)
        )
    return (
        common
        + i.rate * i.strike * math.exp(-i.rate * i.time_to_expiry) * norm.cdf(-d2)
        - i.dividend_yield * i.spot * math.exp(-i.dividend_yield * i.time_to_expiry) * norm.cdf(-d1)
    )


def implied_vol(
    *,
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    right: Right,
    initial_guess: float = 0.20,
    tolerance: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver. Returns IV in annual decimal (e.g. 0.18 = 18%).

    Falls back to bisection if Newton overshoots into negative IV territory.
    """
    if time_to_expiry <= 0 or market_price <= 0:
        return float("nan")

    iv = initial_guess
    for _ in range(max_iter):
        i = BSMInputs(spot, strike, time_to_expiry, rate, dividend_yield, iv, right)
        p = price(i)
        v = vega(i)
        diff = p - market_price
        if abs(diff) < tolerance:
            return iv
        if v < 1e-8:
            break
        new_iv = iv - diff / v
        if new_iv <= 0:
            new_iv = iv / 2
        iv = new_iv

    # Fallback: bisection over [0.001, 5.0]
    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        p_mid = price(BSMInputs(spot, strike, time_to_expiry, rate, dividend_yield, mid, right))
        if abs(p_mid - market_price) < tolerance:
            return mid
        if p_mid > market_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
