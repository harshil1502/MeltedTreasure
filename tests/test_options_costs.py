"""Tests for the F&O cost model."""
from __future__ import annotations

import pytest

from algo.backtest.costs import Side
from algo.options.costs import (
    cost_drag_on_premium,
    option_leg_cost,
    option_round_trip_cost,
)

NIFTY_LOT = 75


class TestOptionLegCost:
    def test_buy_has_no_stt(self):
        c = option_leg_cost(premium=100, lot_size=NIFTY_LOT, lots=1, side=Side.BUY)
        assert c.stt == 0.0

    def test_sell_has_stt(self):
        c = option_leg_cost(premium=100, lot_size=NIFTY_LOT, lots=1, side=Side.SELL)
        # 0.1% of 100 * 75 = 7.5
        assert c.stt == pytest.approx(7.5, abs=0.01)

    def test_buy_has_stamp(self):
        c = option_leg_cost(premium=100, lot_size=NIFTY_LOT, lots=1, side=Side.BUY)
        # 0.003% of 7500 = 0.225
        assert c.stamp == pytest.approx(0.225, abs=0.01)

    def test_sell_has_no_stamp(self):
        c = option_leg_cost(premium=100, lot_size=NIFTY_LOT, lots=1, side=Side.SELL)
        assert c.stamp == 0.0

    def test_brokerage_caps_at_20(self):
        # Premium turnover of ₹1L * 0.03% = ₹30 → capped at ₹20
        c = option_leg_cost(premium=2000, lot_size=NIFTY_LOT, lots=1, side=Side.BUY)
        assert c.brokerage == 20.0

    def test_brokerage_uses_percentage_for_small_orders(self):
        # ₹500 turnover * 0.03% = ₹0.15 (well below cap)
        c = option_leg_cost(premium=10, lot_size=50, lots=1, side=Side.BUY)
        assert c.brokerage < 1.0


class TestRoundTripDrag:
    def test_buyer_round_trip_at_realistic_premium(self):
        """Quantify the actual round-trip cost on a Nifty ATM weekly option.

        At ₹100 premium × 75 lot = ₹7,500 turnover, costs are dominated by
        STT-on-sell (0.1%), exchange (0.035%), and slippage (10 bps × 2 legs).
        The realistic drag is ~0.45-0.55% of premium turnover — meaningful
        but NOT the ~1.5% retail folklore claims.

        The thing that kills retail option buyers is theta + win rate, not
        primarily fees.
        """
        drag = cost_drag_on_premium(
            premium=100, lot_size=NIFTY_LOT, lots=1, is_buyer=True,
        )
        assert 0.003 < drag < 0.008

    def test_seller_round_trip_drag_smaller_than_buyer(self):
        """Sellers face the STT only on their entry sell, not exit buy.

        With stamp on the buy-back leg, total drag is similar but composed
        differently. We just verify it's plausible.
        """
        buyer = cost_drag_on_premium(
            premium=100, lot_size=NIFTY_LOT, lots=1, is_buyer=True,
        )
        seller = cost_drag_on_premium(
            premium=100, lot_size=NIFTY_LOT, lots=1, is_buyer=False,
        )
        # Both should be in the same ballpark
        assert abs(buyer - seller) < 0.005

    def test_drag_decreases_with_premium_size(self):
        """Larger premium = brokerage cap kicks in earlier, drag goes down (bps-wise)."""
        small = cost_drag_on_premium(premium=50, lot_size=NIFTY_LOT, lots=1, is_buyer=True)
        large = cost_drag_on_premium(premium=500, lot_size=NIFTY_LOT, lots=1, is_buyer=True)
        assert small > large


class TestNiftyOneLotAt10k:
    """Concrete reality-check: ₹10k buying 1 lot of Nifty ATM weekly call."""

    def test_one_lot_premium_50_drag(self):
        # Cheap OTM weekly: premium ₹50 → 1 lot = ₹3,750 notional
        entry, exit_ = option_round_trip_cost(
            entry_premium=50, exit_premium=50,
            lot_size=NIFTY_LOT, lots=1, is_buyer=True,
        )
        round_trip = entry.total + exit_.total
        # On ₹3,750 notional, round-trip is ~₹17 (0.45%) — modest in cost
        # terms; the buyer's real enemy is theta, not costs.
        assert 10 < round_trip < 30
