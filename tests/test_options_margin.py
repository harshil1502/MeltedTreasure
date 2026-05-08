"""Tests for margin approximations against published Zerodha numbers."""
from __future__ import annotations

import pytest

from algo.options.margin import (
    NIFTY_LOT,
    Structure,
    estimate,
    lots_that_fit,
)


class TestNakedShort:
    def test_one_lot_at_24k_around_125k(self):
        m = estimate(Structure.NAKED_SHORT, spot=24000, lots=1)
        # Zerodha calc shows ~₹1.2-1.5L for naked short Nifty option
        assert 100_000 < m.margin_inr < 200_000


class TestShortStraddle:
    def test_one_lot_at_24k_in_published_range(self):
        m = estimate(Structure.SHORT_STRADDLE, spot=24000, lots=1)
        # Two legs minus SPAN benefit ≈ ₹1.5-2.5L typical
        assert 150_000 < m.margin_inr < 300_000

    def test_straddle_smaller_than_two_naked(self):
        straddle = estimate(Structure.SHORT_STRADDLE, spot=24000, lots=1).margin_inr
        two_naked = 2 * estimate(Structure.NAKED_SHORT, spot=24000, lots=1).margin_inr
        assert straddle < two_naked


class TestIronCondor:
    def test_50_point_wide_condor_one_lot(self):
        m = estimate(
            Structure.IRON_CONDOR, spot=24000, lots=1,
            spread_width_points=50, net_credit_per_lot=10,
        )
        # Max loss = 50*75 - 10 = ₹3,740 per lot
        assert m.margin_inr == pytest.approx(3740, abs=1)

    def test_condor_capital_efficient(self):
        condor = estimate(
            Structure.IRON_CONDOR, spot=24000, lots=1,
            spread_width_points=100, net_credit_per_lot=20,
        ).margin_inr
        straddle = estimate(Structure.SHORT_STRADDLE, spot=24000, lots=1).margin_inr
        # Condor margin should be a tiny fraction of straddle margin
        assert condor < straddle / 10


class TestCapitalSizing:
    def test_3L_capital_does_not_fit_straddle_with_buffer(self):
        """At ₹3L with 30% safety buffer, usable is ₹2.1L; one straddle
        needs ~₹2.27L → 0 lots. Real-world constraint: at ₹3L you should
        be running defined-risk structures (iron condors), not naked shorts.
        """
        n = lots_that_fit(
            capital_inr=300_000, structure=Structure.SHORT_STRADDLE, spot=24000,
        )
        assert n == 0

    def test_5L_capital_fits_one_short_straddle(self):
        n = lots_that_fit(
            capital_inr=500_000, structure=Structure.SHORT_STRADDLE, spot=24000,
        )
        # ₹3.5L usable / ~₹2.27L per lot = 1 lot
        assert n == 1

    def test_1L_capital_fits_many_iron_condors(self):
        n = lots_that_fit(
            capital_inr=100_000, structure=Structure.IRON_CONDOR, spot=24000,
            spread_width_points=50, net_credit_per_lot=10,
        )
        # ₹70k / ₹3,740 ≈ 18 lots
        assert n >= 15

    def test_buffer_reduces_lot_count(self):
        no_buf = lots_that_fit(
            capital_inr=500_000, structure=Structure.SHORT_STRADDLE,
            spot=24000, safety_buffer=0,
        )
        with_buf = lots_that_fit(
            capital_inr=500_000, structure=Structure.SHORT_STRADDLE,
            spot=24000, safety_buffer=0.30,
        )
        assert with_buf <= no_buf
