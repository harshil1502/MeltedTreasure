"""PaperBroker: accepts orders, fills on the next tick after submission.

Design choices:
- One pending-order queue, processed FIFO.
- Orders submitted at tick T fill on the NEXT incoming snapshot (T+1) at that
  snapshot's quoted price ± slippage. This eliminates look-ahead bias —
  strategies cannot fill at the price they used to make the decision.
- Pricing: prefer real prices from snapshot.option_prices[contract.key()];
  fall back to BSM synthetic from snapshot.spot/iv.
- Costs: full F&O cost model on every fill.
- Position math: weighted-average entry price; realisation on opposite-side fills.
"""
from __future__ import annotations

from typing import Callable, Optional

from algo.backtest.costs import Side
from algo.options.costs import option_leg_cost
from algo.options.synthetic_pricing import synthetic_option_price
from algo.paper.types import (
    Fill,
    MarketSnapshot,
    OptionContract,
    Order,
    OrderId,
    Position,
)

PriceFn = Callable[[OptionContract, MarketSnapshot], float]


def default_price_fn(contract: OptionContract, snap: MarketSnapshot) -> float:
    """Use real prices when present; otherwise BSM synthetic."""
    real = snap.option_prices.get(contract.key())
    if real is not None and real > 0:
        return real
    minutes = max(int((contract.expiry - snap.timestamp).total_seconds() / 60), 1)
    return synthetic_option_price(
        spot=snap.spot, strike=contract.strike,
        minutes_to_expiry=minutes, iv=snap.iv, right=contract.right,
    )


class PaperBroker:
    def __init__(
        self,
        *,
        starting_cash: float,
        slippage_bps: float = 10.0,
        price_fn: PriceFn = default_price_fn,
    ) -> None:
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.slippage_bps = slippage_bps
        self.price_fn = price_fn
        self.positions: dict[str, Position] = {}
        self._pending: list[Order] = []
        self._fills: list[Fill] = []
        self._next_order_id = 1

    # ----- order submission -----

    def submit(
        self, *, contract: OptionContract, side: Side, lots: int,
        now: "pd.Timestamp", tag: str = "",
    ) -> OrderId:
        oid = OrderId(self._next_order_id)
        self._next_order_id += 1
        self._pending.append(Order(
            id=oid, contract=contract, side=side, lots=lots,
            submitted_at=now, tag=tag,
        ))
        return oid

    def cancel(self, order_id: OrderId) -> bool:
        before = len(self._pending)
        self._pending = [o for o in self._pending if o.id != order_id]
        return len(self._pending) < before

    # ----- tick processing -----

    def process_tick(self, snap: MarketSnapshot) -> list[Fill]:
        """Fill any pending orders submitted strictly before this snapshot."""
        fills: list[Fill] = []
        remaining: list[Order] = []
        for order in self._pending:
            if order.submitted_at >= snap.timestamp:
                # Order arrived at-or-after this tick; it must wait for the next.
                remaining.append(order)
                continue
            fills.append(self._execute(order, snap))
        self._pending = remaining
        self._fills.extend(fills)
        return fills

    def _execute(self, order: Order, snap: MarketSnapshot) -> Fill:
        quoted = self.price_fn(order.contract, snap)
        slip = self.slippage_bps / 10_000
        # Buyer pays UP, seller takes DOWN
        if order.side is Side.BUY:
            fill_premium = quoted * (1 + slip)
        else:
            fill_premium = quoted * (1 - slip)
        costs = option_leg_cost(
            premium=fill_premium, lot_size=order.contract.lot_size,
            lots=order.lots, side=order.side, slippage_bps=0,
        )
        fill = Fill(
            order_id=order.id, contract=order.contract, side=order.side,
            lots=order.lots, fill_premium=fill_premium, fill_time=snap.timestamp,
            quoted_premium=quoted, costs=costs, tag=order.tag,
        )
        self._apply_fill(fill)
        return fill

    def _apply_fill(self, fill: Fill) -> None:
        """Apply a fill to cash and position.

        Accounting model:
        - Cash absorbs all cash flows including costs (via fill.cash_flow).
        - realised_pnl tracks gross trading P&L on closed lots only.
          Costs are NOT subtracted from realised_pnl — they're already in cash.
        - equity() = cash + unrealised_MTM(open_positions); this is correct
          because the realised slice has already moved out of open positions
          and into cash via cash_flow.
        """
        self.cash += fill.cash_flow

        key = fill.contract.key()
        pos = self.positions.get(key) or Position(contract=fill.contract)
        lot_size = fill.contract.lot_size

        signed = fill.lots if fill.side is Side.BUY else -fill.lots
        new_lots = pos.lots + signed
        same_direction = pos.lots == 0 or (pos.lots > 0) == (signed > 0)

        if same_direction:
            # Opening or adding: weighted-average entry
            if new_lots != 0:
                pos.avg_entry_premium = (
                    pos.avg_entry_premium * abs(pos.lots)
                    + fill.fill_premium * abs(signed)
                ) / abs(new_lots)
        else:
            # Reducing, closing, or reversing
            closed = min(abs(signed), abs(pos.lots))
            sign = 1 if pos.lots > 0 else -1
            pos.realised_pnl += (
                (fill.fill_premium - pos.avg_entry_premium) * sign * closed * lot_size
            )
            if abs(signed) > abs(pos.lots):
                # Reversed past flat → residual opens at fill price
                pos.avg_entry_premium = fill.fill_premium
            elif new_lots == 0:
                pos.avg_entry_premium = 0.0
            # Reduced but still on same side → avg_entry unchanged

        pos.lots = new_lots
        self.positions[key] = pos

    # ----- reporting -----

    def equity(self, snap: MarketSnapshot) -> float:
        """Cash + unrealised mark-to-market across all open positions."""
        unrealised = 0.0
        for pos in self.positions.values():
            if pos.lots == 0:
                continue
            mark = self.price_fn(pos.contract, snap)
            unrealised += pos.mark_to_market(mark)
        return self.cash + unrealised

    def realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self.positions.values())

    def fills(self) -> list[Fill]:
        return list(self._fills)

    def pending_orders(self) -> list[Order]:
        return list(self._pending)
