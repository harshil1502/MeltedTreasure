"""End-to-end paper session: reuse the ORB strategy through the paper broker.

Purpose: prove the paper layer is wired correctly by running the same logic
that produced our ORB backtest and confirming positions, fills, and PnL flow
through the broker without surprises.

This is NOT additional evidence the strategy works. It's evidence the
EXECUTION INFRASTRUCTURE works. When real Kite data lands, the same
strategy adapter runs unchanged against real ticks.
"""
from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd

from algo.backtest.costs import Side
from algo.data.intraday import load_nifty_intraday
from algo.options.greeks import Right
from algo.paper.broker import PaperBroker
from algo.paper.runner import run_paper_session, snapshots_from_index_bars
from algo.paper.types import MarketSnapshot, OptionContract
from algo.strategies.options_orb import (
    MARKET_OPEN, ORB_END, TIME_STOP, NIFTY_LOT, _next_weekly_expiry,
)
from algo.options.synthetic_pricing import round_strike_to_nifty


class ORBPaperStrategy:
    """ORB rules expressed against the paper-broker callback interface.

    Per-day state machine:
        WAIT_ORB → MONITORING → ENTERED → EXITED
    """
    def __init__(self) -> None:
        self.day_state: dict[pd.Timestamp, dict] = {}

    def _today(self, ts: pd.Timestamp) -> dict:
        d = ts.normalize()
        if d not in self.day_state:
            self.day_state[d] = {
                "phase": "WAIT_ORB",
                "high": float("-inf"),
                "low": float("inf"),
                "contract": None,
                "entry_spot": None,
                "stop_idx": None,
                "target_idx": None,
                "direction": None,
            }
        return self.day_state[d]

    def on_snapshot(self, snap: MarketSnapshot, broker: PaperBroker) -> None:
        ts = snap.timestamp
        local_t = ts.time()
        if local_t < MARKET_OPEN or local_t > TIME_STOP:
            return
        st = self._today(ts)

        # Build opening range
        if local_t <= ORB_END:
            st["high"] = max(st["high"], snap.spot)
            st["low"] = min(st["low"], snap.spot)
            return

        # Past ORB window
        if st["phase"] == "WAIT_ORB":
            st["phase"] = "MONITORING"

        if st["phase"] == "MONITORING":
            if snap.spot > st["high"]:
                st["direction"] = "CALL"
                self._enter(snap, broker, st)
            elif snap.spot < st["low"]:
                st["direction"] = "PUT"
                self._enter(snap, broker, st)
            return

        if st["phase"] == "ENTERED":
            self._maybe_exit(snap, broker, st)

    def _enter(self, snap: MarketSnapshot, broker: PaperBroker, st: dict) -> None:
        right = Right.CALL if st["direction"] == "CALL" else Right.PUT
        strike = round_strike_to_nifty(snap.spot)
        expiry = _next_weekly_expiry(snap.timestamp)
        c = OptionContract(
            underlying="NIFTY", strike=strike, expiry=expiry,
            right=right, lot_size=NIFTY_LOT,
        )
        broker.submit(
            contract=c, side=Side.BUY, lots=1, now=snap.timestamp,
            tag=f"ORB-entry-{st['direction']}",
        )
        st["contract"] = c
        st["entry_spot"] = snap.spot
        rng = st["high"] - st["low"]
        if st["direction"] == "CALL":
            st["stop_idx"] = st["low"]
            st["target_idx"] = snap.spot + 1.5 * rng
        else:
            st["stop_idx"] = st["high"]
            st["target_idx"] = snap.spot - 1.5 * rng
        st["phase"] = "ENTERED"

    def _maybe_exit(self, snap: MarketSnapshot, broker: PaperBroker, st: dict) -> None:
        spot = snap.spot
        local_t = snap.timestamp.time()
        hit_stop = (st["direction"] == "CALL" and spot <= st["stop_idx"]) or \
                   (st["direction"] == "PUT" and spot >= st["stop_idx"])
        hit_target = (st["direction"] == "CALL" and spot >= st["target_idx"]) or \
                     (st["direction"] == "PUT" and spot <= st["target_idx"])
        time_up = local_t >= TIME_STOP

        if hit_stop or hit_target or time_up:
            broker.submit(
                contract=st["contract"], side=Side.SELL, lots=1, now=snap.timestamp,
                tag=f"ORB-exit-{'stop' if hit_stop else 'target' if hit_target else 'time'}",
            )
            st["phase"] = "EXITED"


def main() -> None:
    print("Loading 5m Nifty bars...")
    bars = load_nifty_intraday(interval="5m", days=58)
    print(f"  {len(bars)} bars over {bars.index.normalize().nunique()} trading days")

    broker = PaperBroker(starting_cash=300_000, slippage_bps=10.0)
    strategy = ORBPaperStrategy()

    log_path = Path("data/cache/paper_session_orb.jsonl")
    if log_path.exists():
        log_path.unlink()

    print("\nRunning paper session...")
    summary = run_paper_session(
        snapshots_from_index_bars(bars),
        strategy=strategy, broker=broker,
        log_path=log_path,
        snapshot_log_every=120,
    )

    print(f"\nSnapshots processed: {summary['snapshots']}")
    print(f"Fills:               {summary['fills']}")
    print(f"Realised PnL:        ₹{summary['realised_pnl']:.2f}")
    print(f"Final cash:          ₹{summary['final_cash']:.2f}")
    print(f"Final equity:        ₹{summary['final_equity']:.2f}")
    print(f"Net P&L vs start:    ₹{summary['final_equity'] - 300_000:+.2f}")
    print(f"\nLog: {log_path}")
    print(f"Log size: {log_path.stat().st_size / 1024:.1f} KB")

    # Sample the log
    print("\nFirst few log lines:")
    with log_path.open() as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
