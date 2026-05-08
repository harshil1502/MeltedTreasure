"""Cost-model sanity checks.

These pin down the numbers we expect for the ₹10k account so a future code change
can't silently inflate backtest returns. If Zerodha or SEBI changes a rate, these
tests should be updated alongside the constants in `algo.backtest.costs`.
"""
from __future__ import annotations

import pytest

from algo.backtest.costs import (
    BROKERAGE_CAP_MIS,
    DP_CHARGE_PER_SCRIP,
    Product,
    Side,
    cost_drag_pct,
    leg_costs,
    round_trip_cost,
)


class TestMisIntraday:
    def test_small_order_brokerage_is_percentage(self):
        # ₹10k turnover × 0.03% = ₹3, well below the ₹20 cap
        leg = leg_costs(
            price=100.0, quantity=100, product=Product.MIS, side=Side.BUY, slippage_bps=0
        )
        assert leg.brokerage == pytest.approx(3.0)

    def test_large_order_brokerage_capped(self):
        leg = leg_costs(
            price=1000.0, quantity=100, product=Product.MIS, side=Side.BUY, slippage_bps=0
        )
        assert leg.brokerage == pytest.approx(BROKERAGE_CAP_MIS)

    def test_stt_only_on_sell(self):
        buy = leg_costs(
            price=100, quantity=100, product=Product.MIS, side=Side.BUY, slippage_bps=0
        )
        sell = leg_costs(
            price=100, quantity=100, product=Product.MIS, side=Side.SELL, slippage_bps=0
        )
        assert buy.stt == 0.0
        assert sell.stt == pytest.approx(10_000 * 0.00025)

    def test_no_dp_charge_on_intraday(self):
        sell = leg_costs(
            price=100,
            quantity=100,
            product=Product.MIS,
            side=Side.SELL,
            is_first_sell_of_scrip_today=True,
            slippage_bps=0,
        )
        assert sell.dp_charge == 0.0


class TestCncDelivery:
    def test_zero_brokerage(self):
        leg = leg_costs(
            price=500, quantity=20, product=Product.CNC, side=Side.BUY, slippage_bps=0
        )
        assert leg.brokerage == 0.0

    def test_dp_charge_applied_once(self):
        first_sell = leg_costs(
            price=500,
            quantity=20,
            product=Product.CNC,
            side=Side.SELL,
            is_first_sell_of_scrip_today=True,
            slippage_bps=0,
        )
        same_day_again = leg_costs(
            price=500,
            quantity=20,
            product=Product.CNC,
            side=Side.SELL,
            is_first_sell_of_scrip_today=False,
            slippage_bps=0,
        )
        assert first_sell.dp_charge == pytest.approx(DP_CHARGE_PER_SCRIP)
        assert same_day_again.dp_charge == 0.0


class TestRoundTripDrag:
    def test_intraday_10k_round_trip_drag_under_1pct(self):
        # Realistic ₹10k single-name intraday round trip with 5 bps slippage each side
        drag = cost_drag_pct(
            entry_price=100.0,
            exit_price=100.0,
            quantity=100,
            product=Product.MIS,
            slippage_bps=5.0,
        )
        # Brokerage 3+3, STT 2.5 on sell, exchange ~0.65, sebi ~0.02, stamp 0.3,
        # gst on (broker+exch+sebi), slippage 5 each side. ~₹17 on ₹10k notional.
        assert 0.10 < drag < 0.30

    def test_cnc_swing_drag_dominated_by_dp_at_small_size(self):
        buy, sell = round_trip_cost(
            entry_price=200.0,
            exit_price=210.0,
            quantity=50,  # ₹10k notional
            product=Product.CNC,
            slippage_bps=5.0,
        )
        # DP charge alone is ~₹16 — that's ~16 bps of ₹10k. Massive at small size.
        assert sell.dp_charge == pytest.approx(DP_CHARGE_PER_SCRIP)
        assert (buy.total + sell.total) > sell.dp_charge  # other costs add on top

    def test_drag_decreases_with_notional_under_brokerage_cap(self):
        # When brokerage cap binds, larger notionals dilute fixed costs
        small = cost_drag_pct(
            entry_price=100, exit_price=100, quantity=100, product=Product.MIS, slippage_bps=0
        )
        large = cost_drag_pct(
            entry_price=1000, exit_price=1000, quantity=1000, product=Product.MIS, slippage_bps=0
        )
        assert large < small
