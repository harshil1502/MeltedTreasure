"""Approximate SPAN + Exposure margin for Indian F&O.

Disclaimer: This is a *rough* approximation of Zerodha's actual margin
requirements. Real SPAN runs 16 worst-case scenarios on the entire portfolio;
this module gives ballpark numbers good enough for capital-sizing decisions.
Always cross-check with Zerodha's margin calculator (zerodha.com/margin-calculator)
before going live.

References (verified Nov 2025):
- Short single-leg Nifty option: ~7-9% of notional ≈ ₹1.2-1.5L/lot at Nifty 24k
- Short straddle (call + put, same strike): margin benefit, ~₹1.5-1.8L total
- Short strangle (OTM call + OTM put): similar to straddle, slightly less
- Iron condor (defined risk, 4 legs): margin = (spread width × lot_size)
- Bull put / bear call credit spread: margin = (spread width × lot_size) - net credit

Indian SPAN methodology details:
https://www.nseindia.com/products-services/equity-derivatives-margins
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

NIFTY_LOT = 75
BANKNIFTY_LOT = 30  # post-2024 lot size

# Empirical fractions of notional for naked short option margin.
# Higher in high-VIX regimes. Use 9% as a slightly conservative default.
SHORT_OPTION_MARGIN_FRAC = 0.09

# Margin benefit when both call and put are short at same strike (straddle)
# or different strikes (strangle): SPAN partially offsets.
STRADDLE_MARGIN_DISCOUNT = 0.30  # ~30% reduction vs sum of two naked legs


class Structure(str, Enum):
    NAKED_SHORT = "naked_short"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"


@dataclass(frozen=True)
class MarginEstimate:
    structure: Structure
    margin_inr: float
    notional_inr: float
    margin_pct_of_notional: float


def naked_short_margin(*, spot: float, lot_size: int, lots: int) -> float:
    return SHORT_OPTION_MARGIN_FRAC * spot * lot_size * lots


def short_straddle_margin(*, spot: float, lot_size: int, lots: int) -> float:
    # Two naked legs minus SPAN benefit
    naked_two = 2 * naked_short_margin(spot=spot, lot_size=lot_size, lots=lots)
    return naked_two * (1 - STRADDLE_MARGIN_DISCOUNT)


def short_strangle_margin(*, spot: float, lot_size: int, lots: int) -> float:
    # Approximately same as straddle in first-order; slight reduction with
    # wider strikes ignored here.
    return short_straddle_margin(spot=spot, lot_size=lot_size, lots=lots)


def iron_condor_margin(
    *, spread_width_points: float, lot_size: int, lots: int, net_credit_per_lot: float = 0.0,
) -> float:
    """Margin for a defined-risk iron condor.

    Max loss = spread_width × lot_size − net_credit_received. That's the
    capital at risk and equals the margin block.
    """
    return max(spread_width_points * lot_size - net_credit_per_lot, 0) * lots


def credit_spread_margin(
    *, spread_width_points: float, lot_size: int, lots: int, net_credit_per_lot: float = 0.0,
) -> float:
    """Margin for a bull put or bear call credit spread."""
    return iron_condor_margin(
        spread_width_points=spread_width_points,
        lot_size=lot_size, lots=lots, net_credit_per_lot=net_credit_per_lot,
    )


def estimate(
    structure: Structure,
    *,
    spot: float,
    lot_size: int = NIFTY_LOT,
    lots: int = 1,
    spread_width_points: float = 0,
    net_credit_per_lot: float = 0.0,
) -> MarginEstimate:
    """Single entry point for sizing decisions."""
    if structure is Structure.NAKED_SHORT:
        m = naked_short_margin(spot=spot, lot_size=lot_size, lots=lots)
    elif structure is Structure.SHORT_STRADDLE:
        m = short_straddle_margin(spot=spot, lot_size=lot_size, lots=lots)
    elif structure is Structure.SHORT_STRANGLE:
        m = short_strangle_margin(spot=spot, lot_size=lot_size, lots=lots)
    elif structure is Structure.IRON_CONDOR:
        m = iron_condor_margin(
            spread_width_points=spread_width_points, lot_size=lot_size,
            lots=lots, net_credit_per_lot=net_credit_per_lot,
        )
    elif structure is Structure.CREDIT_SPREAD:
        m = credit_spread_margin(
            spread_width_points=spread_width_points, lot_size=lot_size,
            lots=lots, net_credit_per_lot=net_credit_per_lot,
        )
    else:
        raise ValueError(f"unknown structure {structure}")

    notional = spot * lot_size * lots
    return MarginEstimate(
        structure=structure, margin_inr=m, notional_inr=notional,
        margin_pct_of_notional=m / notional if notional else 0,
    )


def lots_that_fit(
    capital_inr: float,
    structure: Structure,
    *,
    spot: float,
    lot_size: int = NIFTY_LOT,
    spread_width_points: float = 0,
    net_credit_per_lot: float = 0.0,
    safety_buffer: float = 0.30,
) -> int:
    """Max lots that fit in capital, leaving a safety buffer for variation
    margin calls. Default 30% buffer means we use only 70% of stated capital
    for position margin.
    """
    usable = capital_inr * (1 - safety_buffer)
    one_lot = estimate(
        structure, spot=spot, lot_size=lot_size, lots=1,
        spread_width_points=spread_width_points,
        net_credit_per_lot=net_credit_per_lot,
    ).margin_inr
    if one_lot <= 0:
        return 0
    return int(usable // one_lot)
