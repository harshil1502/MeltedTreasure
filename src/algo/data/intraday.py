"""Intraday Nifty index loader via yfinance.

yfinance availability for ^NSEI (Nifty 50 index):
    1m  → last 7 days only
    5m  → last 60 days
    15m → last 60 days
    30m → last 60 days
    1h  → last 730 days

For ORB-style intraday research we use 5m. 60 days is statistically thin —
at most ~60 ORB events. Use real Kite Connect intraday data for production.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from algo.data.loaders import _ensure_cache

NIFTY_INDEX = "^NSEI"
INDIA_VIX = "^INDIAVIX"


def load_nifty_intraday(
    interval: str = "5m",
    days: int = 60,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch Nifty 50 intraday OHLCV. Returns DataFrame indexed by tz-aware timestamp."""
    import yfinance as yf

    if interval not in {"1m", "5m", "15m", "30m", "1h"}:
        raise ValueError(f"unsupported interval {interval}")
    # yfinance enforces a strict 60-day cap on intraday intervals; cap here
    # to avoid an off-by-one when callers ask for 60.
    days = min(days, 58)
    end = date.today()
    start = end - timedelta(days=days)
    cache: Path = _ensure_cache() / f"nifty_{interval}_{start}_{end}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = yf.download(
        tickers=NIFTY_INDEX, start=str(start), end=str(end),
        interval=interval, progress=False, auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {NIFTY_INDEX} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    # Normalize index to Asia/Kolkata so 9:15 means 9:15 IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.to_parquet(cache)
    return df
