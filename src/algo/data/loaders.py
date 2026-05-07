"""Daily OHLCV loaders for Indian equities.

Two free sources are wired:
- yfinance: convenient, adjusted prices, broad coverage, occasional gaps
- NSE bhavcopy: authoritative end-of-day from the exchange, unadjusted

For backtests prefer yfinance adjusted Close (handles splits + dividends).
For execution-matching realism use bhavcopy unadjusted prices and apply
corporate actions explicitly.
"""
from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/cache")
NSE_BHAVCOPY_URL = (
    "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}


def _ensure_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def load_yfinance(
    symbols: list[str],
    start: str | date,
    end: str | date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV for one or more `.NS` symbols via yfinance.

    Returns a DataFrame indexed by Date with a MultiIndex column (field, symbol).
    """
    import yfinance as yf

    end = end or date.today().isoformat()
    cache = _ensure_cache() / f"yf_{'_'.join(sorted(symbols))[:60]}_{start}_{end}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = yf.download(
        tickers=symbols,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {symbols}")
    df.to_parquet(cache)
    return df


def load_nse_bhavcopy(on: date) -> pd.DataFrame:
    """Fetch a single day's NSE bhavcopy. Returns DataFrame; raises on miss."""
    cache = _ensure_cache() / f"bhav_{on.isoformat()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = NSE_BHAVCOPY_URL.format(ddmmyyyy=on.strftime("%d%m%Y"))
    resp = requests.get(url, headers=NSE_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    df.columns = [c.strip() for c in df.columns]
    df["DATE"] = pd.to_datetime(df[" DATE1"].str.strip(), format="%d-%b-%Y")
    df.to_parquet(cache)
    return df


def adjusted_close_panel(
    symbols: list[str], start: str | date, end: str | date | None = None
) -> pd.DataFrame:
    """Convenience: returns a (date × symbol) frame of adjusted Close prices."""
    df = load_yfinance(symbols, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        adj = df["Adj Close"].copy()
    else:
        adj = df[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
    adj.index = pd.to_datetime(adj.index)
    return adj.sort_index()
