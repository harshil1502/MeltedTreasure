"""Idempotency tests so daily workflow runs don't pollute the audit trail."""
from __future__ import annotations

from datetime import date

from algo.live.portfolio import PaperPortfolio


def test_mark_same_day_replaces_in_place():
    p = PaperPortfolio()
    p.mark_to_market({}, at=date(2026, 5, 7))
    p.mark_to_market({}, at=date(2026, 5, 7))
    assert len(p.equity_history) == 1


def test_mark_new_day_appends():
    p = PaperPortfolio()
    p.mark_to_market({}, at=date(2026, 5, 7))
    p.mark_to_market({}, at=date(2026, 5, 8))
    assert len(p.equity_history) == 2


def test_record_signal_same_day_replaces_weights():
    p = PaperPortfolio()
    p.record_signal(date(2026, 5, 7), {"NIFTYBEES": 1.0})
    p.record_signal(date(2026, 5, 7), {"GOLDBEES": 1.0})
    assert len(p.signals) == 1
    assert p.signals[0]["weights"] == {"GOLDBEES": 1.0}


def test_record_signal_different_days_appends():
    p = PaperPortfolio()
    p.record_signal(date(2026, 4, 30), {"NIFTYBEES": 1.0})
    p.record_signal(date(2026, 5, 30), {"GOLDBEES": 1.0})
    assert len(p.signals) == 2
