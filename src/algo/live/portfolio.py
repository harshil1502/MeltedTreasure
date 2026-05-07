"""Paper portfolio state for forward simulation.

State is a JSON file so it's auditable, diff-able and easy to back up.
There is no DB and no broker integration — the user records fills manually
after placing trades through Zerodha's web UI (or skips fills entirely
to track the strategy's signal history without any real capital).

State schema:
{
    "initial_capital": 10000.0,
    "cash": 4321.5,
    "positions": {"NIFTYBEES": {"qty": 12, "avg_cost": 234.5}},
    "signals": [
        {"date": "2026-01-31", "weights": {"GOLDBEES": 1.0}}, ...
    ],
    "fills": [
        {"date": "2026-02-01", "symbol": "GOLDBEES", "side": "BUY",
         "qty": 60, "price": 81.5, "cost_total": 12.4}, ...
    ],
    "equity_history": [{"date": "2026-01-31", "equity": 10000.0}, ...]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from algo.backtest.costs import Product, Side, leg_costs

DEFAULT_STATE_PATH = Path("state/paper_portfolio.json")


@dataclass
class PaperPortfolio:
    initial_capital: float = 10_000.0
    cash: float = 10_000.0
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    signals: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    equity_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "PaperPortfolio":
        if not path.exists():
            p = cls()
            p.save(path)
            return p
        data = json.loads(path.read_text())
        return cls(**data)

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))

    def record_signal(self, signal_date: date, weights: dict[str, float]) -> None:
        self.signals.append(
            {"date": signal_date.isoformat(), "weights": weights}
        )

    def record_fill(
        self,
        *,
        fill_date: date,
        symbol: str,
        side: Side,
        qty: int,
        price: float,
        product: Product = Product.CNC,
        slippage_bps: float = 5.0,
    ) -> dict[str, Any]:
        if qty <= 0 or price <= 0:
            raise ValueError(f"qty and price must be positive (got qty={qty}, price={price})")
        is_first_sell_today = side is Side.SELL and not any(
            f["date"] == fill_date.isoformat() and f["symbol"] == symbol and f["side"] == "SELL"
            for f in self.fills
        )
        costs = leg_costs(
            price=price, quantity=qty, product=product, side=side,
            slippage_bps=slippage_bps,
            is_first_sell_of_scrip_today=is_first_sell_today,
        )
        notional = price * qty
        if side is Side.BUY:
            outflow = notional + costs.total
            if outflow > self.cash:
                raise ValueError(
                    f"insufficient cash: need ₹{outflow:.2f}, have ₹{self.cash:.2f}"
                )
            self.cash -= outflow
            pos = self.positions.setdefault(symbol, {"qty": 0, "avg_cost": 0.0})
            new_qty = pos["qty"] + qty
            pos["avg_cost"] = (pos["avg_cost"] * pos["qty"] + notional) / new_qty
            pos["qty"] = new_qty
        else:  # SELL
            pos = self.positions.get(symbol)
            if not pos or pos["qty"] < qty:
                raise ValueError(
                    f"cannot sell {qty} of {symbol}: holding "
                    f"{pos['qty'] if pos else 0}"
                )
            self.cash += notional - costs.total
            pos["qty"] -= qty
            if pos["qty"] == 0:
                self.positions.pop(symbol)
        fill_record = {
            "date": fill_date.isoformat(),
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "price": price,
            "cost_total": round(costs.total, 2),
        }
        self.fills.append(fill_record)
        return fill_record

    def mark_to_market(self, prices: dict[str, float], at: date) -> float:
        equity = self.cash + sum(
            prices.get(sym, 0.0) * pos["qty"]
            for sym, pos in self.positions.items()
        )
        self.equity_history.append({"date": at.isoformat(), "equity": round(equity, 2)})
        return equity
