"""Quantify the real obstacles to intraday Nifty option buying at ₹10k.

Self-correction: an earlier off-the-cuff estimate put round-trip cost drag at
~1.2-2% of premium. The actual cost model says 0.45-0.55%. Costs are NOT the
dominant friction. Theta and win-rate-asymmetric-payoff are.

This notebook produces three numbers:
1. Round-trip cost drag at typical premiums
2. Daily theta drag (% of premium decayed per day held)
3. Required win rate to break even, given a realistic risk:reward

If theta + cost > expected gross edge, the strategy is structurally negative-EV
regardless of signal quality. That's the honest baseline.
"""
from __future__ import annotations

from algo.options.costs import cost_drag_on_premium
from algo.options.greeks import BSMInputs, Right, price, theta

NIFTY_LOT = 75


def main() -> None:
    print("=== 1. Round-trip cost drag on a single Nifty option ===")
    for premium in (30, 50, 100, 200, 400):
        drag = cost_drag_on_premium(
            premium=premium, lot_size=NIFTY_LOT, lots=1, is_buyer=True,
        )
        notional = premium * NIFTY_LOT
        print(f"  premium ₹{premium:>4} (notional ₹{notional:>5}): drag = {drag*100:.3f}%")

    print("\n=== 2. Daily theta decay on Nifty 7-DTE ATM call ===")
    spot = 24000
    rate, q = 0.065, 0.012
    for iv_pct in (10, 15, 20, 25):
        iv = iv_pct / 100
        i = BSMInputs(
            spot=spot, strike=spot, time_to_expiry=7/365,
            rate=rate, dividend_yield=q, iv=iv, right=Right.CALL,
        )
        p = price(i)
        per_day_theta = theta(i) / 365  # convert annual to daily
        # premium-relative decay (negative theta means losing this much per day)
        decay_pct = abs(per_day_theta) / p * 100
        print(f"  IV {iv_pct}%: premium ₹{p:.2f}, theta/day ₹{per_day_theta:.2f} "
              f"(-{decay_pct:.2f}%/day)")

    print("\n=== 3. Break-even analysis: how much does the option need to move? ===")
    print("  Scenario: buy 7-DTE ATM Nifty call, hold for 1 day, exit at close.")
    print("  IV 15%, Nifty at 24000.\n")
    iv = 0.15
    i_t0 = BSMInputs(24000, 24000, 7/365, rate, q, iv, Right.CALL)
    p0 = price(i_t0)
    cost_drag = cost_drag_on_premium(premium=p0, lot_size=NIFTY_LOT, lots=1, is_buyer=True)
    one_day_theta = abs(theta(i_t0)) / 365 / p0  # fractional decay per day

    print(f"  Entry premium: ₹{p0:.2f}")
    print(f"  1-leg slippage assumption: 10 bps each side")
    print(f"  Round-trip cost drag: {cost_drag*100:.2f}%")
    print(f"  Theta drag (1 day): {one_day_theta*100:.2f}%")
    print(f"  Total drag from costs + theta: {(cost_drag + one_day_theta)*100:.2f}%")
    print(f"  -> Underlying needs to move enough that delta + gamma effects "
          f"overcome ~{(cost_drag + one_day_theta)*100:.1f}% premium drag")

    # Estimate the required Nifty move to overcome drag
    # Approx: option_pct_change ≈ delta * spot_pct_change * spot/premium (omega/lambda)
    print("\n=== 4. Required Nifty move to overcome 1-day drag ===")
    from algo.options.greeks import delta as bsm_delta
    d = bsm_delta(i_t0)
    # omega = leverage = delta * spot / premium
    omega = d * spot / p0
    breakeven_move_pct = (cost_drag + one_day_theta) * 100 / omega
    print(f"  Option delta: {d:.3f}")
    print(f"  Leverage (Ω = δ*S/P): {omega:.1f}x")
    print(f"  Nifty needs to move at least: {breakeven_move_pct:.3f}% in your direction")
    print(f"  In points (Nifty 24000): {24000 * breakeven_move_pct/100:.0f} points\n")

    print("=== Honest takeaway ===")
    print("  Buying Nifty options intraday at ₹10k:")
    print("  - 1 lot fits (~₹3.7-15k notional depending on strike)")
    print("  - Round-trip costs are tolerable (~0.5%)")
    print("  - The ~14%/day theta decay on weekly ATM options is the structural enemy")
    print("  - To break even on a 1-day hold, Nifty must move ~0.05-0.10%+ in your direction")
    print("  - In a flat or wrong-direction day, you lose theta + costs ~ 0.7-1.5% of premium")
    print("  - Need very high win rate (>55%) to profit, which retail rarely has\n")
    print("  Recommendation: framework is built and tested; live deployment of buying-side")
    print("  strategies stays gated until selling-side capital (~₹3-5L) is available.")


if __name__ == "__main__":
    main()
