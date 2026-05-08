"""Run the ORB strategy on Nifty intraday 5m bars (last ~60 trading days).

Statistical caveats up front:
- 60 days is too few for confident inference. Treat the result as a
  sanity check on the framework, not a verdict on the strategy.
- Synthetic option pricing uses constant 15% IV. Real IV varies; results
  with skew + smile would differ on OTM strikes (we trade ATM only).
- Stops are evaluated at bar high/low (worst-case fill). No gap modeling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from algo.data.intraday import load_nifty_intraday
from algo.strategies.options_orb import ORBParams, run_orb_backtest

OUT_DIR = Path("data/cache")


def main() -> None:
    print("Fetching Nifty 5m bars (last ~60 days)...")
    bars = load_nifty_intraday(interval="5m", days=60)
    print(f"  {len(bars)} bars from {bars.index.min()} to {bars.index.max()}")
    n_days = bars.index.normalize().nunique()
    print(f"  {n_days} trading days available")

    print("\nRunning ORB backtest (IV=15%, target=1.5x range, ₹10k cap, 10 bps slippage)...")
    trades = run_orb_backtest(bars, ORBParams())

    if not trades:
        print("\nNo trades produced — strategy never found a breakout that fit ₹10k budget.")
        return

    df = pd.DataFrame([{
        "date": t.date.date(),
        "direction": t.direction,
        "entry": t.entry_time.strftime("%H:%M"),
        "exit": t.exit_time.strftime("%H:%M"),
        "reason": t.exit_reason,
        "spot_entry": round(t.spot_entry, 2),
        "spot_exit": round(t.spot_exit, 2),
        "strike": t.strike,
        "premium_in": round(t.premium_entry, 2),
        "premium_out": round(t.premium_exit, 2),
        "pnl_gross": round(t.pnl_gross, 2),
        "cost": round(t.cost, 2),
        "pnl_net": round(t.pnl_net, 2),
    } for t in trades])

    print(f"\n{len(df)} trades over {n_days} trading days "
          f"({len(df)/n_days*100:.0f}% participation rate)")
    print(df.to_string(index=False))

    print("\n=== Aggregate ===")
    pnl_net = df["pnl_net"]
    wins = (pnl_net > 0).sum()
    losses = (pnl_net <= 0).sum()
    print(f"  Win / Loss: {wins} / {losses}  (win rate {wins/len(df)*100:.1f}%)")
    print(f"  Total gross PnL: ₹{df['pnl_gross'].sum():.2f}")
    print(f"  Total cost:      ₹{df['cost'].sum():.2f}")
    print(f"  Total net PnL:   ₹{pnl_net.sum():.2f}")
    print(f"  Avg net trade:   ₹{pnl_net.mean():.2f}")
    print(f"  Best trade:      ₹{pnl_net.max():.2f}")
    print(f"  Worst trade:     ₹{pnl_net.min():.2f}")

    # Required win rate to break even on the observed avg-win/avg-loss:
    avg_win = pnl_net[pnl_net > 0].mean() if wins > 0 else 0
    avg_loss = pnl_net[pnl_net <= 0].mean() if losses > 0 else 0
    if avg_loss < 0:
        # E[trade] = p*W + (1-p)*L = 0  =>  p = -L / (W - L)
        breakeven_p = -avg_loss / (avg_win - avg_loss) * 100
        print(f"  Avg win / avg loss: ₹{avg_win:.2f} / ₹{avg_loss:.2f}")
        print(f"  Break-even win rate at this R:R: {breakeven_p:.1f}%")
        print(f"  Realised win rate: {wins/len(df)*100:.1f}%")
        verdict = "PASSES" if (wins/len(df)*100) > breakeven_p else "FAILS"
        print(f"  Verdict: {verdict} the break-even bar (n={len(df)} too small to be confident)")

    # Equity curve assuming each trade uses ₹10k freshly (no compounding,
    # since at ₹10k we can hold 1 lot total)
    equity = 10_000 + pnl_net.cumsum()
    print(f"\n  Final paper equity: ₹{equity.iloc[-1]:.2f} "
          f"({(equity.iloc[-1]/10000-1)*100:+.2f}%)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "orb_options_trades.parquet")


if __name__ == "__main__":
    main()
