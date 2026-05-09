from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]

    data["return_1d"] = close.pct_change()
    data["log_return_1d"] = np.log(close / close.shift(1))
    data["volume_change"] = data["Volume"].pct_change()

    for lag in [1, 2, 3, 5, 10]:
        data[f"return_lag_{lag}"] = data["return_1d"].shift(lag)

    for window in [5, 10, 20, 50]:
        data[f"ma_{window}"] = close.rolling(window).mean()
        data[f"price_to_ma_{window}"] = close / data[f"ma_{window}"] - 1

    data["rolling_volatility_10"] = data["return_1d"].rolling(10).std()
    data["rolling_volatility_20"] = data["return_1d"].rolling(20).std()
    data["rsi_14"] = _rsi(close, 14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std()
    data["bb_upper"] = rolling_mean_20 + 2 * rolling_std_20
    data["bb_lower"] = rolling_mean_20 - 2 * rolling_std_20
    data["bb_position"] = (close - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])

    data["target"] = (close.shift(-1) > close).astype(int)
    return data.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "target"}
    return [col for col in df.columns if col not in excluded]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
