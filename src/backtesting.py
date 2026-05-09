from __future__ import annotations

import numpy as np
import pandas as pd


def run_directional_backtest(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = predictions.copy()
    result["market_return"] = result["return_1d"]
    result["strategy_signal"] = result["prediction"].replace({0: -1, 1: 1})
    result["strategy_return"] = result["strategy_signal"].shift(1) * result["market_return"]
    result = result.dropna().reset_index(drop=True)

    result["market_equity"] = (1 + result["market_return"]).cumprod()
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()

    metrics = {
        "cumulative_market_return": result["market_equity"].iloc[-1] - 1,
        "cumulative_strategy_return": result["strategy_equity"].iloc[-1] - 1,
        "sharpe_ratio": _sharpe_ratio(result["strategy_return"]),
        "max_drawdown": _max_drawdown(result["strategy_equity"]),
        "win_rate": (result["strategy_return"] > 0).mean(),
    }
    return result, metrics


def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())
