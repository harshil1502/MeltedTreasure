"""Forward-paper CLI.

Subcommands:
    signal           generate today's target allocation, optionally record it
    confirm          record a manual fill into paper state
    status           print current paper portfolio + recent signals
    mark             mark-to-market today and append to equity history
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from algo.backtest.costs import Side
from algo.live.portfolio import DEFAULT_STATE_PATH, PaperPortfolio
from algo.live.signal import generate_current_signal


def _load(state_path: Path) -> PaperPortfolio:
    return PaperPortfolio.load(state_path)


def cmd_signal(args: argparse.Namespace) -> None:
    sig = generate_current_signal()
    print(f"As of: {sig.as_of}")
    print(f"Last rebalance signal date: {sig.last_rebalance_date}")
    print(f"Today is a rebalance day: {sig.is_rebalance_day}")
    print("\nTarget allocation:")
    if not sig.target_weights:
        print("  (cash)")
    for sym, w in sorted(sig.target_weights.items(), key=lambda kv: -kv[1]):
        last_px = sig.underlying_prices.get(sym, 0.0)
        print(f"  {sym:12s}  weight={w:.2%}   last_close=₹{last_px:.2f}")

    if args.record:
        portfolio = _load(args.state)
        portfolio.record_signal(sig.last_rebalance_date or sig.as_of, sig.target_weights)
        portfolio.save(args.state)
        print(f"\nSignal recorded to {args.state}")


def cmd_confirm(args: argparse.Namespace) -> None:
    portfolio = _load(args.state)
    fill_date = date.fromisoformat(args.date) if args.date else date.today()
    side = Side(args.side.upper())
    record = portfolio.record_fill(
        fill_date=fill_date,
        symbol=args.symbol,
        side=side,
        qty=args.qty,
        price=args.price,
    )
    portfolio.save(args.state)
    print(f"Recorded fill: {record}")
    print(f"Cash after: ₹{portfolio.cash:.2f}")
    print(f"Positions: {portfolio.positions}")


def cmd_status(args: argparse.Namespace) -> None:
    portfolio = _load(args.state)
    print(f"Initial capital: ₹{portfolio.initial_capital:.2f}")
    print(f"Cash:            ₹{portfolio.cash:.2f}")
    if portfolio.positions:
        print("\nPositions:")
        for sym, pos in portfolio.positions.items():
            print(f"  {sym:12s}  qty={pos['qty']}  avg_cost=₹{pos['avg_cost']:.2f}")
    else:
        print("\nPositions: (none)")

    if portfolio.signals:
        print(f"\nLast 5 signals:")
        for s in portfolio.signals[-5:]:
            print(f"  {s['date']}  {s['weights']}")

    if portfolio.fills:
        print(f"\nLast 5 fills:")
        for f in portfolio.fills[-5:]:
            print(
                f"  {f['date']}  {f['side']:4s} {f['qty']:>4d} {f['symbol']:12s} "
                f"@ ₹{f['price']:.2f}  cost=₹{f['cost_total']:.2f}"
            )

    if portfolio.equity_history:
        last = portfolio.equity_history[-1]
        print(f"\nLast mark: {last['date']} -> ₹{last['equity']:.2f}")


def cmd_mark(args: argparse.Namespace) -> None:
    sig = generate_current_signal()
    portfolio = _load(args.state)
    eq = portfolio.mark_to_market(sig.underlying_prices, at=sig.as_of)
    portfolio.save(args.state)
    print(f"Marked at {sig.as_of}: equity = ₹{eq:.2f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="algo-paper")
    p.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH,
                   help="path to paper portfolio JSON state")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_signal = sub.add_parser("signal", help="generate current target allocation")
    s_signal.add_argument("--record", action="store_true",
                          help="append the signal to portfolio state")
    s_signal.set_defaults(func=cmd_signal)

    s_confirm = sub.add_parser("confirm", help="record a manual fill into paper state")
    s_confirm.add_argument("--symbol", required=True)
    s_confirm.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"])
    s_confirm.add_argument("--qty", required=True, type=int)
    s_confirm.add_argument("--price", required=True, type=float)
    s_confirm.add_argument("--date", help="ISO date (default: today)")
    s_confirm.set_defaults(func=cmd_confirm)

    s_status = sub.add_parser("status", help="print paper portfolio status")
    s_status.set_defaults(func=cmd_status)

    s_mark = sub.add_parser("mark", help="mark-to-market and append equity point")
    s_mark.set_defaults(func=cmd_mark)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
