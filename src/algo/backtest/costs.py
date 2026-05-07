"""Indian equity transaction cost model (Zerodha retail).

Numbers reflect SEBI/exchange/state stamp duty rules effective FY25 and Zerodha's
public pricing. Verify on https://zerodha.com/charges before going live —
exchange transaction charges and stamp duty have changed multiple times.

Two product types are modeled:
- MIS  (intraday): brokerage applies, STT only on sell side, lower stamp
- CNC  (delivery): zero brokerage, STT both sides, DP charge on sell — the
                   silent killer for small accounts

Slippage is applied as a configurable bps haircut and is NOT a cost in the
fee-and-tax sense; it is realized by entering buys above and sells below the
mid. Both are returned so PnL accounting is honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Product(str, Enum):
    MIS = "MIS"  # intraday
    CNC = "CNC"  # delivery


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp: float
    gst: float
    dp_charge: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.stamp
            + self.gst
            + self.dp_charge
            + self.slippage
        )


# Rates expressed as fractions of turnover unless stated otherwise.
BROKERAGE_RATE_MIS = 0.0003          # 0.03%
BROKERAGE_CAP_MIS = 20.0             # ₹20 per executed order
STT_MIS_SELL = 0.00025               # 0.025% on sell
STT_CNC_BOTH = 0.001                 # 0.1% both sides
EXCHANGE_TXN_NSE = 0.0000322         # 0.00322%
SEBI_RATE = 0.000001                 # 0.0001 per crore == 1e-6 of turnover
STAMP_BUY_MIS = 0.00003              # 0.003% on buy
STAMP_BUY_CNC = 0.00015              # 0.015% on buy
GST_RATE = 0.18
DP_CHARGE_PER_SCRIP = 15.93          # ₹13.5 + 18% GST, charged once per scrip per day on CNC sell


def _brokerage(turnover: float, product: Product) -> float:
    if product is Product.CNC:
        return 0.0
    return min(turnover * BROKERAGE_RATE_MIS, BROKERAGE_CAP_MIS)


def _stt(turnover: float, product: Product, side: Side) -> float:
    if product is Product.CNC:
        return turnover * STT_CNC_BOTH
    if side is Side.SELL:
        return turnover * STT_MIS_SELL
    return 0.0


def _stamp(turnover: float, product: Product, side: Side) -> float:
    if side is not Side.BUY:
        return 0.0
    rate = STAMP_BUY_CNC if product is Product.CNC else STAMP_BUY_MIS
    return turnover * rate


def leg_costs(
    *,
    price: float,
    quantity: int,
    product: Product,
    side: Side,
    slippage_bps: float = 5.0,
    is_first_sell_of_scrip_today: bool = False,
) -> CostBreakdown:
    """Compute charges for one leg (one order fill).

    `is_first_sell_of_scrip_today` toggles the CNC DP charge — it applies once
    per scrip per day on sell, regardless of order count. Pass True for the
    first sell leg of each scrip in your simulation.
    """
    turnover = price * quantity
    brokerage = _brokerage(turnover, product)
    stt = _stt(turnover, product, side)
    exchange = turnover * EXCHANGE_TXN_NSE
    sebi = turnover * SEBI_RATE
    stamp = _stamp(turnover, product, side)
    gst = (brokerage + exchange + sebi) * GST_RATE
    dp = (
        DP_CHARGE_PER_SCRIP
        if (product is Product.CNC and side is Side.SELL and is_first_sell_of_scrip_today)
        else 0.0
    )
    slippage = turnover * (slippage_bps / 10_000)
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange,
        sebi=sebi,
        stamp=stamp,
        gst=gst,
        dp_charge=dp,
        slippage=slippage,
    )


def round_trip_cost(
    *,
    entry_price: float,
    exit_price: float,
    quantity: int,
    product: Product,
    slippage_bps: float = 5.0,
) -> tuple[CostBreakdown, CostBreakdown]:
    """Convenience: cost breakdown for both legs of a round trip.

    Assumes the exit sell is the first (and likely only) sell of the scrip
    that day, so DP charge is applied once for CNC.
    """
    buy = leg_costs(
        price=entry_price,
        quantity=quantity,
        product=product,
        side=Side.BUY,
        slippage_bps=slippage_bps,
    )
    sell = leg_costs(
        price=exit_price,
        quantity=quantity,
        product=product,
        side=Side.SELL,
        slippage_bps=slippage_bps,
        is_first_sell_of_scrip_today=True,
    )
    return buy, sell


def cost_drag_pct(
    *,
    entry_price: float,
    exit_price: float,
    quantity: int,
    product: Product,
    slippage_bps: float = 5.0,
) -> float:
    """Total round-trip cost as a percentage of entry notional.

    Useful for the gut-check: 'is the strategy's edge bigger than this number?'
    """
    buy, sell = round_trip_cost(
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        product=product,
        slippage_bps=slippage_bps,
    )
    notional = entry_price * quantity
    return (buy.total + sell.total) / notional * 100
