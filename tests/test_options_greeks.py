"""Tests for Black-Scholes pricing, Greeks, and IV solver."""
from __future__ import annotations

import pytest

from algo.options.greeks import BSMInputs, Right, delta, gamma, implied_vol, price, theta, vega


def make(spot=24000, strike=24000, dte_days=7, iv=0.15, right=Right.CALL):
    return BSMInputs(
        spot=spot, strike=strike,
        time_to_expiry=dte_days / 365,
        rate=0.065, dividend_yield=0.012,
        iv=iv, right=right,
    )


class TestPrice:
    def test_atm_call_price_positive(self):
        p = price(make())
        assert 50 < p < 500  # plausible range for Nifty 7-DTE ATM at 15% IV

    def test_put_call_parity(self):
        c = price(make(right=Right.CALL))
        p = price(make(right=Right.PUT))
        # C - P = S*e^-qT - K*e^-rT  (for European w/ continuous q)
        i = make()
        import math
        expected = (
            i.spot * math.exp(-i.dividend_yield * i.time_to_expiry)
            - i.strike * math.exp(-i.rate * i.time_to_expiry)
        )
        assert (c - p) == pytest.approx(expected, abs=0.5)

    def test_otm_call_cheaper_than_itm(self):
        otm = price(make(strike=25000))
        itm = price(make(strike=23000))
        assert otm < itm

    def test_at_expiry_call_returns_intrinsic(self):
        p = price(make(spot=24500, strike=24000, dte_days=0))
        assert p == pytest.approx(500.0)

    def test_at_expiry_put_returns_intrinsic(self):
        p = price(make(spot=23500, strike=24000, dte_days=0, right=Right.PUT))
        assert p == pytest.approx(500.0)


class TestGreeks:
    def test_atm_call_delta_near_half(self):
        d = delta(make())
        assert 0.45 < d < 0.55

    def test_atm_put_delta_near_neg_half(self):
        d = delta(make(right=Right.PUT))
        assert -0.55 < d < -0.45

    def test_gamma_positive_atm(self):
        g = gamma(make())
        assert g > 0

    def test_theta_negative_for_long(self):
        t = theta(make())
        assert t < 0  # long options decay

    def test_vega_positive(self):
        v = vega(make())
        assert v > 0

    def test_short_dated_atm_has_higher_gamma(self):
        short_dte = gamma(make(dte_days=2))
        long_dte = gamma(make(dte_days=30))
        assert short_dte > long_dte


class TestImpliedVol:
    def test_recover_iv_from_price_round_trip(self):
        i = make(iv=0.17)
        market_p = price(i)
        recovered = implied_vol(
            market_price=market_p,
            spot=i.spot, strike=i.strike,
            time_to_expiry=i.time_to_expiry,
            rate=i.rate, dividend_yield=i.dividend_yield,
            right=i.right,
        )
        assert recovered == pytest.approx(0.17, abs=1e-3)

    def test_recover_iv_for_high_vol(self):
        i = make(iv=0.45)
        market_p = price(i)
        recovered = implied_vol(
            market_price=market_p,
            spot=i.spot, strike=i.strike,
            time_to_expiry=i.time_to_expiry,
            rate=i.rate, dividend_yield=i.dividend_yield,
            right=i.right,
        )
        assert recovered == pytest.approx(0.45, abs=1e-3)
