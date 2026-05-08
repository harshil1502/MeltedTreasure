"""Paper broker tests: no-lookahead, position math, costs, MTM."""
from __future__ import annotations

import pandas as pd
import pytest

from algo.backtest.costs import Side
from algo.options.greeks import Right
from algo.paper.broker import PaperBroker
from algo.paper.types import MarketSnapshot, OptionContract


def contract(strike: int = 24000, right: Right = Right.CALL) -> OptionContract:
    return OptionContract(
        underlying="NIFTY", strike=strike,
        expiry=pd.Timestamp("2026-05-14 15:30", tz="Asia/Kolkata"),
        right=right, lot_size=75,
    )


def snap(ts: str, spot: float = 24000, iv: float = 0.15) -> MarketSnapshot:
    return MarketSnapshot(timestamp=pd.Timestamp(ts, tz="Asia/Kolkata"), spot=spot, iv=iv)


class TestNoLookahead:
    def test_order_at_T_does_not_fill_at_T(self):
        b = PaperBroker(starting_cash=500_000)
        s1 = snap("2026-05-08 09:30")
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        # Same-timestamp tick: must NOT fill
        fills = b.process_tick(s1)
        assert fills == []
        assert len(b.pending_orders()) == 1

    def test_order_at_T_fills_at_T_plus_1(self):
        b = PaperBroker(starting_cash=500_000)
        s1 = snap("2026-05-08 09:30")
        s2 = snap("2026-05-08 09:35", spot=24050)
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        fills = b.process_tick(s2)
        assert len(fills) == 1
        assert fills[0].fill_time == s2.timestamp


class TestSlippageAndCosts:
    def test_buy_pays_slip_above_quote(self):
        b = PaperBroker(starting_cash=500_000, slippage_bps=10)
        s1 = snap("2026-05-08 09:30")
        s2 = snap("2026-05-08 09:35")
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        f = b.process_tick(s2)[0]
        # Slippage must move buyer's fill ABOVE quoted
        assert f.fill_premium > f.quoted_premium

    def test_sell_takes_slip_below_quote(self):
        b = PaperBroker(starting_cash=500_000, slippage_bps=10)
        s1 = snap("2026-05-08 09:30")
        s2 = snap("2026-05-08 09:35")
        b.submit(contract=contract(), side=Side.SELL, lots=1, now=s1.timestamp)
        f = b.process_tick(s2)[0]
        assert f.fill_premium < f.quoted_premium

    def test_buy_costs_reduce_cash(self):
        b = PaperBroker(starting_cash=500_000)
        s1 = snap("2026-05-08 09:30")
        s2 = snap("2026-05-08 09:35")
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        f = b.process_tick(s2)[0]
        notional = f.fill_premium * f.contract.lot_size * f.lots
        # Cash decreases by exactly premium + costs
        assert b.cash == pytest.approx(500_000 - notional - f.costs.total, abs=0.01)


class TestPositionMath:
    def test_open_long_one_lot(self):
        b = PaperBroker(starting_cash=500_000)
        s1, s2 = snap("2026-05-08 09:30"), snap("2026-05-08 09:35")
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        pos = b.positions[contract().key()]
        assert pos.lots == 1
        assert pos.avg_entry_premium > 0

    def test_scale_long_weighted_avg(self):
        b = PaperBroker(starting_cash=2_000_000)
        # Two buys at different premiums
        s1 = snap("2026-05-08 09:30", spot=24000)
        s2 = snap("2026-05-08 09:35", spot=24000)  # fill at this tick
        s3 = snap("2026-05-08 10:00", spot=24300)  # different price now
        s4 = snap("2026-05-08 10:05", spot=24300)
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        first_premium = b.positions[contract().key()].avg_entry_premium
        b.submit(contract=contract(), side=Side.BUY, lots=2, now=s3.timestamp)
        b.process_tick(s4)
        pos = b.positions[contract().key()]
        assert pos.lots == 3
        # New avg should be biased toward the larger position (2 lots at higher price)
        assert pos.avg_entry_premium > first_premium

    def test_close_long_realises_pnl(self):
        b = PaperBroker(starting_cash=2_000_000)
        s1 = snap("2026-05-08 09:30", spot=24000)
        s2 = snap("2026-05-08 09:35", spot=24000)
        s3 = snap("2026-05-08 10:00", spot=24300)  # spot higher → call worth more
        s4 = snap("2026-05-08 10:05", spot=24300)
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        entry = b.positions[contract().key()].avg_entry_premium
        b.submit(contract=contract(), side=Side.SELL, lots=1, now=s3.timestamp)
        b.process_tick(s4)
        pos = b.positions[contract().key()]
        assert pos.lots == 0
        # Realised PnL should be positive (call gained value as spot rose)
        assert pos.realised_pnl > 0

    def test_reverse_past_flat(self):
        b = PaperBroker(starting_cash=2_000_000)
        s1 = snap("2026-05-08 09:30")
        s2 = snap("2026-05-08 09:35")
        s3 = snap("2026-05-08 10:00", spot=24100)
        s4 = snap("2026-05-08 10:05", spot=24100)
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        # Sell 3 lots → close 1, open 2 short
        b.submit(contract=contract(), side=Side.SELL, lots=3, now=s3.timestamp)
        b.process_tick(s4)
        pos = b.positions[contract().key()]
        assert pos.lots == -2
        # avg_entry_premium for the new short = the fill premium of the reversal
        assert pos.avg_entry_premium > 0


class TestEquity:
    def test_equity_equals_cash_when_flat(self):
        b = PaperBroker(starting_cash=500_000)
        s = snap("2026-05-08 09:30")
        assert b.equity(s) == 500_000

    def test_equity_tracks_mtm(self):
        b = PaperBroker(starting_cash=2_000_000)
        s1 = snap("2026-05-08 09:30", spot=24000)
        s2 = snap("2026-05-08 09:35", spot=24000)
        s3 = snap("2026-05-08 10:00", spot=24300)  # call up
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        eq_at_entry = b.equity(s2)
        eq_with_move = b.equity(s3)
        # Should have gained some MTM as spot rose
        assert eq_with_move > eq_at_entry

    def test_realised_plus_unrealised_consistent_after_close(self):
        b = PaperBroker(starting_cash=2_000_000)
        s1 = snap("2026-05-08 09:30", spot=24000)
        s2 = snap("2026-05-08 09:35", spot=24000)
        s3 = snap("2026-05-08 10:00", spot=24300)
        s4 = snap("2026-05-08 10:05", spot=24300)
        b.submit(contract=contract(), side=Side.BUY, lots=1, now=s1.timestamp)
        b.process_tick(s2)
        b.submit(contract=contract(), side=Side.SELL, lots=1, now=s3.timestamp)
        b.process_tick(s4)
        pos = b.positions[contract().key()]
        # Equity = starting_cash + realised_pnl - all_costs
        all_costs = sum(f.costs.total for f in b.fills())
        expected_eq = 2_000_000 + pos.realised_pnl - all_costs
        assert b.equity(s4) == pytest.approx(expected_eq, abs=0.01)


class TestPriceFnFallback:
    def test_uses_real_price_when_present(self):
        b = PaperBroker(starting_cash=500_000)
        s1 = snap("2026-05-08 09:30")
        c = contract()
        # Provide a real price that's NOT what BSM would return
        custom = MarketSnapshot(
            timestamp=pd.Timestamp("2026-05-08 09:35", tz="Asia/Kolkata"),
            spot=24000, iv=0.15,
            option_prices={c.key(): 333.33},
        )
        b.submit(contract=c, side=Side.BUY, lots=1, now=s1.timestamp)
        f = b.process_tick(custom)[0]
        # quoted should be the provided price (slippage applies on top)
        assert f.quoted_premium == pytest.approx(333.33, abs=0.01)
