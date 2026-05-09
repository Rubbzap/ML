from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename_axis("Date").reset_index()
    data["Ticker"] = ticker.upper()
    return data
