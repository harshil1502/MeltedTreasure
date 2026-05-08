"""Indian F&O transaction cost model — Zerodha retail.

Reference: https://zerodha.com/charges (verify before going live; F&O STT was
hiked Oct 2024 as part of Budget 2024-25, effective rates updated below).

Options:
    Brokerage:   ₹20 or 0.03% per executed order, whichever lower
    STT:         0.1% on premium on SELL side only (was 0.0625% pre-Oct 2024)
    Exchange:    NSE 0.03503% on premium (both sides)
    SEBI:        ₹10 per crore = 1e-6 of turnover (both sides)
    Stamp duty:  0.003% on premium on BUY side only (one-time)
    GST:         18% on (brokerage + exchange + SEBI)
    DP charges:  none (options aren't held in demat)

Futures:
    STT:         0.02% on SELL side (was 0.0125% pre-Oct 2024)
    Exchange:    NSE 0.0019% on turnover (both sides)
    Other:       same structure as options
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from algo.backtest.costs import Side


class FNOInstrument(str, Enum):
    OPTION = "OPT"
    FUTURE = "FUT"


@dataclass(frozen=True)
class FNOCostBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage + self.stt + self.exchange_txn + self.sebi
            + self.stamp + self.gst + self.slippage
        )


# Rates (decimals).
BROKERAGE_RATE = 0.0003                   # 0.03%
BROKERAGE_CAP = 20.0                      # ₹20 per executed order

STT_OPTION_SELL = 0.001                   # 0.1% on premium, sell side (post Oct 2024)
STT_FUTURE_SELL = 0.0002                  # 0.02% on turnover, sell side (post Oct 2024)

EXCHANGE_TXN_OPTION = 0.0003503           # 0.03503% on premium
EXCHANGE_TXN_FUTURE = 0.000019            # 0.0019% on turnover

SEBI_RATE = 0.000001                      # 0.0001% (₹10 per crore)
STAMP_OPTION_BUY = 0.00003                # 0.003% on premium, buy side
STAMP_FUTURE_BUY = 0.00002                # 0.002% on turnover, buy side
GST_RATE = 0.18


def option_leg_cost(
    *,
    premium: float,
    lot_size: int,
    lots: int,
    side: Side,
    slippage_bps: float = 10.0,
) -> FNOCostBreakdown:
    """Costs for one option leg.

    Notional for STT/exchange/stamp/SEBI is the premium turnover:
        turnover = premium × lot_size × lots
    """
    if lots <= 0 or lot_size <= 0:
        raise ValueError("lots and lot_size must be positive")
    turnover = premium * lot_size * lots
    brokerage = min(turnover * BROKERAGE_RATE, BROKERAGE_CAP)
    stt = turnover * STT_OPTION_SELL if side is Side.SELL else 0.0
    exchange = turnover * EXCHANGE_TXN_OPTION
    sebi = turnover * SEBI_RATE
    stamp = turnover * STAMP_OPTION_BUY if side is Side.BUY else 0.0
    gst = (brokerage + exchange + sebi) * GST_RATE
    slippage = turnover * (slippage_bps / 10_000)
    return FNOCostBreakdown(
        brokerage=brokerage, stt=stt, exchange_txn=exchange,
        sebi=sebi, stamp=stamp, gst=gst, slippage=slippage,
    )


def option_round_trip_cost(
    *,
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int,
    is_buyer: bool,
    slippage_bps: float = 10.0,
) -> tuple[FNOCostBreakdown, FNOCostBreakdown]:
    """Costs for an entry + exit pair.

    is_buyer=True  : long option (BUY then SELL)
    is_buyer=False : short option (SELL then BUY)  — note this needs SPAN margin
                     not modeled here.
    """
    entry_side = Side.BUY if is_buyer else Side.SELL
    exit_side = Side.SELL if is_buyer else Side.BUY
    entry = option_leg_cost(
        premium=entry_premium, lot_size=lot_size, lots=lots,
        side=entry_side, slippage_bps=slippage_bps,
    )
    exit_ = option_leg_cost(
        premium=exit_premium, lot_size=lot_size, lots=lots,
        side=exit_side, slippage_bps=slippage_bps,
    )
    return entry, exit_


def cost_drag_on_premium(
    *,
    premium: float,
    lot_size: int,
    lots: int,
    is_buyer: bool,
    slippage_bps: float = 10.0,
) -> float:
    """Round-trip cost as a fraction of entry premium (assuming exit ≈ entry).

    Useful for the 'how much does the option need to move just to break even'
    gut-check.
    """
    entry, exit_ = option_round_trip_cost(
        entry_premium=premium, exit_premium=premium,
        lot_size=lot_size, lots=lots, is_buyer=is_buyer,
        slippage_bps=slippage_bps,
    )
    notional = premium * lot_size * lots
    return (entry.total + exit_.total) / notional
