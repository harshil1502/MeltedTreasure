"""Standard performance metrics for an equity curve.

Daily-frequency returns are assumed (Indian equities trade ~252 days/year).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Metrics:
    total_return_pct: float
    cagr_pct: float
    annualized_vol_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    n_trades: int
    win_rate_pct: float
    avg_trade_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> Metrics:
    if equity.empty or len(equity) < 2:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years = len(equity) / TRADING_DAYS_PER_YEAR
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    excess = rets - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    sharpe = float(excess.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    n_trades = 0
    win_rate = 0.0
    avg_trade = 0.0
    if trades is not None and not trades.empty:
        # Pair buy/sell legs per symbol in FIFO order to compute realised PnL
        pnls: list[float] = []
        for sym, grp in trades.sort_values("date").groupby("symbol"):
            buy_qty = 0
            buy_cost = 0.0
            buy_notional = 0.0
            for _, t in grp.iterrows():
                if t["side"] == "BUY":
                    buy_qty += t["qty"]
                    buy_notional += t["notional"]
                    buy_cost += t["cost_total"]
                else:  # SELL
                    if buy_qty <= 0:
                        continue
                    sell_qty = min(t["qty"], buy_qty)
                    avg_buy_px = buy_notional / buy_qty
                    pnl = (t["price"] - avg_buy_px) * sell_qty - t["cost_total"] - (
                        buy_cost * (sell_qty / buy_qty)
                    )
                    pnls.append(pnl / (avg_buy_px * sell_qty))
                    buy_qty -= sell_qty
                    buy_notional -= avg_buy_px * sell_qty
                    buy_cost -= buy_cost * (sell_qty / max(buy_qty + sell_qty, 1))
        if pnls:
            n_trades = len(pnls)
            win_rate = float(np.mean([p > 0 for p in pnls])) * 100
            avg_trade = float(np.mean(pnls)) * 100

    return Metrics(
        total_return_pct=total * 100,
        cagr_pct=cagr * 100,
        annualized_vol_pct=vol * 100,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd * 100,
        calmar=calmar,
        n_trades=n_trades,
        win_rate_pct=win_rate,
        avg_trade_pct=avg_trade,
    )
