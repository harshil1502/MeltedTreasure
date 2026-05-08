"""Position sizing for a small Indian retail account.

At ₹10k, whole-share constraints dominate. A stock priced at ₹3,500 (e.g. RELIANCE)
permits at most 2 shares, blowing concentration limits. This module returns
integer share counts that respect:
- max % of capital per position
- whole-share rounding
- minimum-trade economics (skip if cost-drag > expected edge)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SizingResult:
    quantity: int
    notional: float
    capital_used_pct: float
    skipped_reason: str | None = None


def fixed_fractional(
    *,
    capital: float,
    price: float,
    fraction: float = 0.33,
    min_notional: float = 2_000.0,
) -> SizingResult:
    """Allocate `fraction` of capital to one position, rounded to whole shares.

    Skips the trade if rounded notional falls below `min_notional` — at ₹10k,
    a position smaller than ~₹2k carries cost drag that exceeds typical edge.
    """
    target_notional = capital * fraction
    qty = math.floor(target_notional / price)
    if qty <= 0:
        return SizingResult(0, 0.0, 0.0, skipped_reason="price exceeds budget")
    notional = qty * price
    if notional < min_notional:
        return SizingResult(qty, notional, notional / capital * 100, skipped_reason="below min notional")
    return SizingResult(qty, notional, notional / capital * 100)


def atr_stop_quantity(
    *,
    capital: float,
    price: float,
    atr: float,
    risk_per_trade_pct: float = 1.0,
    atr_multiple: float = 2.0,
) -> SizingResult:
    """Volatility-targeted sizing: risk a fixed % of capital to the ATR stop.

    quantity = (capital × risk_per_trade) / (atr × atr_multiple)
    Caps at 50% of capital to avoid concentration on low-vol names.
    """
    risk_budget = capital * (risk_per_trade_pct / 100)
    stop_distance = atr * atr_multiple
    if stop_distance <= 0:
        return SizingResult(0, 0.0, 0.0, skipped_reason="invalid ATR")
    raw_qty = risk_budget / stop_distance
    qty = math.floor(raw_qty)
    if qty <= 0:
        return SizingResult(0, 0.0, 0.0, skipped_reason="risk budget below 1 share")
    notional = qty * price
    cap = capital * 0.5
    if notional > cap:
        qty = math.floor(cap / price)
        notional = qty * price
    return SizingResult(qty, notional, notional / capital * 100)
