from __future__ import annotations

import argparse
import json

import joblib

from config import MODEL_DIR, RANDOM_STATE
from data_loader import download_stock_data
from features import add_technical_indicators, get_feature_columns
from modeling import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock movement forecasting model.")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker, e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--start", default="2015-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["logistic", "random_forest", "xgboost"],
        help="Model type",
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_stock_data(args.ticker, args.start, args.end)
    dataset = add_technical_indicators(raw)
    feature_columns = get_feature_columns(dataset)
    result = train_and_evaluate(
        dataset,
        feature_columns,
        args.model,
        tune=args.tune,
        random_state=RANDOM_STATE,
    )

    model_path = MODEL_DIR / f"{args.ticker.upper()}_{args.model}.joblib"
    joblib.dump(
        {
            "model": result.estimator,
            "feature_columns": feature_columns,
            "metrics": result.metrics,
        },
        model_path,
    )

    print(f"Saved model: {model_path}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
