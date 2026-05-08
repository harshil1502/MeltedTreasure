"""Driver loop: feed snapshots → broker fills → strategy submits → log."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import pandas as pd

from algo.paper.broker import PaperBroker
from algo.paper.types import Fill, MarketSnapshot


class Strategy(Protocol):
    def on_snapshot(self, snap: MarketSnapshot, broker: PaperBroker) -> None: ...


def _serialize_fill(f: Fill) -> dict:
    return {
        "order_id": f.order_id.value,
        "contract": f.contract.key(),
        "side": f.side.value,
        "lots": f.lots,
        "fill_time": f.fill_time.isoformat(),
        "fill_premium": round(f.fill_premium, 4),
        "quoted_premium": round(f.quoted_premium, 4),
        "cost_total": round(f.costs.total, 2),
        "cash_flow": round(f.cash_flow, 2),
        "tag": f.tag,
    }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **fields) -> None:
        self._fh.write(json.dumps({"event": event, **fields}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def run_paper_session(
    snapshots: Iterable[MarketSnapshot],
    *,
    strategy: Strategy,
    broker: PaperBroker,
    log_path: Optional[Path] = None,
    snapshot_log_every: int = 60,
) -> dict:
    """Run a paper session over a stream of snapshots.

    Order of operations per snapshot:
      1. Process tick → fill any orders submitted before this timestamp
      2. Call strategy → may submit new orders for next tick
      3. Log fills + (optionally) periodic equity snapshot
    """
    logger = JsonlLogger(log_path) if log_path else None
    last_equity_log_idx = -1
    final_equity: float = broker.starting_cash
    final_snap: Optional[MarketSnapshot] = None
    snap_count = 0

    try:
        for i, snap in enumerate(snapshots):
            snap_count = i + 1
            final_snap = snap
            fills = broker.process_tick(snap)
            if logger is not None:
                for f in fills:
                    logger.write("fill", **_serialize_fill(f))
            strategy.on_snapshot(snap, broker)
            if logger is not None and (i - last_equity_log_idx) >= snapshot_log_every:
                logger.write(
                    "equity",
                    timestamp=snap.timestamp.isoformat(),
                    cash=round(broker.cash, 2),
                    equity=round(broker.equity(snap), 2),
                    realised=round(broker.realised_pnl(), 2),
                    open_positions=sum(1 for p in broker.positions.values() if p.lots != 0),
                )
                last_equity_log_idx = i
        if final_snap is not None:
            final_equity = broker.equity(final_snap)
    finally:
        if logger is not None:
            logger.write(
                "session_end",
                snapshots=snap_count,
                final_cash=round(broker.cash, 2),
                final_equity=round(final_equity, 2),
                realised=round(broker.realised_pnl(), 2),
                fills=len(broker.fills()),
            )
            logger.close()

    return {
        "snapshots": snap_count,
        "final_equity": final_equity,
        "final_cash": broker.cash,
        "realised_pnl": broker.realised_pnl(),
        "fills": len(broker.fills()),
    }


def snapshots_from_index_bars(
    bars: pd.DataFrame, iv: float = 0.15,
) -> Iterable[MarketSnapshot]:
    """Convert an OHLCV DataFrame (indexed by timestamp) into snapshots
    using the bar Close as the prevailing spot. One snapshot per bar.
    """
    for ts, row in bars.iterrows():
        yield MarketSnapshot(
            timestamp=pd.Timestamp(ts), spot=float(row["Close"]), iv=iv,
        )
