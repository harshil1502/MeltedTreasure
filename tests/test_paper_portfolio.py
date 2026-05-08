from __future__ import annotations

from datetime import date

import pytest

from algo.backtest.costs import Side
from algo.live.portfolio import PaperPortfolio


def test_initial_state_is_all_cash():
    p = PaperPortfolio()
    assert p.cash == 10_000.0
    assert p.positions == {}
    assert p.equity_history == []


def test_buy_decrements_cash_and_creates_position(tmp_path):
    p = PaperPortfolio()
    record = p.record_fill(
        fill_date=date(2026, 5, 1),
        symbol="NIFTYBEES",
        side=Side.BUY,
        qty=40,
        price=200.0,
    )
    assert p.cash < 10_000.0  # spent on buy + costs
    assert p.cash > 1_000.0   # didn't blow up
    assert p.positions["NIFTYBEES"]["qty"] == 40
    assert p.positions["NIFTYBEES"]["avg_cost"] == pytest.approx(200.0)
    assert record["side"] == "BUY"


def test_sell_credits_cash_and_clears_position():
    p = PaperPortfolio()
    p.record_fill(fill_date=date(2026, 5, 1), symbol="GOLDBEES",
                  side=Side.BUY, qty=100, price=80.0)
    p.record_fill(fill_date=date(2026, 6, 1), symbol="GOLDBEES",
                  side=Side.SELL, qty=100, price=85.0)
    assert "GOLDBEES" not in p.positions
    # Net should be roughly 100 * (85-80) = 500 minus costs and DP charge
    assert 9_400 < p.cash < 10_500


def test_cannot_oversell():
    p = PaperPortfolio()
    p.record_fill(fill_date=date(2026, 5, 1), symbol="GOLDBEES",
                  side=Side.BUY, qty=10, price=80.0)
    with pytest.raises(ValueError, match="cannot sell"):
        p.record_fill(fill_date=date(2026, 5, 2), symbol="GOLDBEES",
                      side=Side.SELL, qty=11, price=82.0)


def test_cannot_overspend():
    p = PaperPortfolio(initial_capital=1_000.0, cash=1_000.0)
    with pytest.raises(ValueError, match="insufficient cash"):
        p.record_fill(fill_date=date(2026, 5, 1), symbol="NIFTYBEES",
                      side=Side.BUY, qty=100, price=200.0)


def test_mark_to_market_includes_positions_at_current_price():
    p = PaperPortfolio()
    p.record_fill(fill_date=date(2026, 5, 1), symbol="NIFTYBEES",
                  side=Side.BUY, qty=40, price=200.0)
    cash_after_buy = p.cash
    eq = p.mark_to_market({"NIFTYBEES": 210.0}, at=date(2026, 5, 31))
    assert eq == pytest.approx(cash_after_buy + 40 * 210.0)
    assert p.equity_history[-1]["equity"] == pytest.approx(eq, abs=0.01)


def test_state_roundtrip(tmp_path):
    state_path = tmp_path / "p.json"
    p = PaperPortfolio()
    p.record_fill(fill_date=date(2026, 5, 1), symbol="NIFTYBEES",
                  side=Side.BUY, qty=40, price=200.0)
    p.record_signal(date(2026, 5, 31), {"NIFTYBEES": 1.0})
    p.save(state_path)

    p2 = PaperPortfolio.load(state_path)
    assert p2.cash == p.cash
    assert p2.positions == p.positions
    assert p2.signals == p.signals
    assert p2.fills == p.fills
