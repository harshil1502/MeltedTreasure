"""Trading universes.

NIFTY_50 is the current snapshot. For backtests, use a point-in-time constituents
file (NSE publishes index reconstitution events) — survivorship bias materially
inflates momentum-strategy returns.
"""
from __future__ import annotations

NIFTY_50: tuple[str, ...] = (
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZYDUSLIFE",
)

ETF_ROTATION: tuple[str, ...] = (
    "NIFTYBEES",      # Nifty 50 equity
    "JUNIORBEES",     # Nifty Next 50 equity
    "GOLDBEES",       # Gold
    "LIQUIDBEES",     # Liquid debt proxy
)


def to_yfinance(symbols: tuple[str, ...]) -> list[str]:
    """Append `.NS` suffix for yfinance NSE quotes."""
    return [f"{s}.NS" for s in symbols]
