"""Daily long-only portfolio backtest with realistic Indian-equity costs.

The engine is deliberately small and inspectable:
- Daily rebalance to target weights from a strategy
- Whole-share rounding (matters at ₹10k)
- Cost model applied per leg (buy + sell), including DP charge once per
  scrip per day on CNC sells
- Marks the portfolio to close-of-day prices

Limitations (deliberate, document them):
- No intraday fills, no partial fills, no slippage beyond the bps assumption
- No short selling
- No interest on cash
- Survivorship bias in any backtest where universe == current Nifty 50
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algo.backtest.costs import CostBreakdown, Product, Side, leg_costs


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    product: Product = Product.CNC
    slippage_bps: float = 5.0
    min_position_inr: float = 2_000.0  # skip trades smaller than this


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.DataFrame  # date × symbol, share count
    cash: pd.Series
    trades: pd.DataFrame    # one row per fill
    cost_total: float
    config: BacktestConfig = field(default=None)  # type: ignore[assignment]


def _record_trade(
    trades: list[dict],
    *,
    date: pd.Timestamp,
    symbol: str,
    side: Side,
    qty: int,
    price: float,
    costs: CostBreakdown,
) -> None:
    trades.append(
        {
            "date": date,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "price": price,
            "notional": price * qty,
            "cost_total": costs.total,
            "cost_brokerage": costs.brokerage,
            "cost_stt": costs.stt,
            "cost_exchange": costs.exchange_txn,
            "cost_sebi": costs.sebi,
            "cost_stamp": costs.stamp,
            "cost_gst": costs.gst,
            "cost_dp": costs.dp_charge,
            "cost_slippage": costs.slippage,
        }
    )


def run_backtest(
    *,
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate a daily long-only portfolio.

    Args:
        prices: (date × symbol) frame of close prices, no NaNs in a row that has weights.
        target_weights: (date × symbol) frame of target weights in [0, 1] summing to ≤ 1.
                        Rows present in this frame trigger a rebalance on that close.
                        Rows absent => hold prior day's positions.
        config: backtest knobs.

    Returns BacktestResult.
    """
    cfg = config or BacktestConfig()
    prices = prices.sort_index().ffill()
    target_weights = target_weights.reindex(prices.index)  # forward-align

    cash = cfg.initial_capital
    positions = pd.Series(0, index=prices.columns, dtype=int)
    equity_curve = []
    positions_history = []
    cash_history = []
    trades: list[dict] = []
    total_cost = 0.0

    rebalance_dates = target_weights.dropna(how="all").index

    for date, row in prices.iterrows():
        # 1. Mark current portfolio at today's close
        portfolio_value = cash + float((positions * row).sum())

        # 2. Rebalance if today is a rebalance day
        if date in rebalance_dates:
            target = target_weights.loc[date].fillna(0.0)
            target_value = portfolio_value * target
            target_qty = (target_value / row).fillna(0).astype(int)

            # Process sells first to free up cash
            sells = (positions - target_qty).where(lambda s: s > 0, 0)
            for sym, qty_to_sell in sells.items():
                qty_to_sell = int(qty_to_sell)
                if qty_to_sell <= 0:
                    continue
                price = float(row[sym])
                if not np.isfinite(price) or price <= 0:
                    continue
                costs = leg_costs(
                    price=price,
                    quantity=qty_to_sell,
                    product=cfg.product,
                    side=Side.SELL,
                    slippage_bps=cfg.slippage_bps,
                    is_first_sell_of_scrip_today=True,
                )
                cash += price * qty_to_sell - costs.total
                positions[sym] -= qty_to_sell
                total_cost += costs.total
                _record_trade(trades, date=date, symbol=sym, side=Side.SELL,
                              qty=qty_to_sell, price=price, costs=costs)

            # Then buys, sized against current cash
            buys = (target_qty - positions).where(lambda s: s > 0, 0)
            buy_targets = [(sym, int(q)) for sym, q in buys.items() if int(q) > 0]
            for sym, qty_to_buy in buy_targets:
                price = float(row[sym])
                if not np.isfinite(price) or price <= 0:
                    continue
                # Skip dust positions — cost drag eats them
                if price * qty_to_buy < cfg.min_position_inr:
                    continue
                # Affordability check (rough; cost added below)
                affordable = int(min(qty_to_buy, max(0, cash // price)))
                if affordable <= 0:
                    continue
                costs = leg_costs(
                    price=price,
                    quantity=affordable,
                    product=cfg.product,
                    side=Side.BUY,
                    slippage_bps=cfg.slippage_bps,
                )
                outflow = price * affordable + costs.total
                if outflow > cash:
                    # Trim by one share if cost pushed us over
                    affordable -= 1
                    if affordable <= 0:
                        continue
                    costs = leg_costs(
                        price=price,
                        quantity=affordable,
                        product=cfg.product,
                        side=Side.BUY,
                        slippage_bps=cfg.slippage_bps,
                    )
                    outflow = price * affordable + costs.total
                cash -= outflow
                positions[sym] += affordable
                total_cost += costs.total
                _record_trade(trades, date=date, symbol=sym, side=Side.BUY,
                              qty=affordable, price=price, costs=costs)

        # 3. Mark equity
        eq = cash + float((positions * row).sum())
        equity_curve.append(eq)
        cash_history.append(cash)
        positions_history.append(positions.copy())

    equity = pd.Series(equity_curve, index=prices.index, name="equity")
    cash_s = pd.Series(cash_history, index=prices.index, name="cash")
    pos_df = pd.DataFrame(positions_history, index=prices.index)
    trades_df = pd.DataFrame(trades)
    return BacktestResult(
        equity=equity,
        positions=pos_df,
        cash=cash_s,
        trades=trades_df,
        cost_total=total_cost,
        config=cfg,
    )
