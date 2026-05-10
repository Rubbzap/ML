from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class TrainResult:
    model_name: str
    estimator: object
    metrics: dict
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame | None


def time_based_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return (
        train[feature_columns],
        test[feature_columns],
        train["target"],
        test["target"],
    )


def build_model(model_name: str, random_state: int = 42):
    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )

    raise ValueError(f"Unknown model: {model_name}")


def tune_model(model_name: str, x_train, y_train, random_state: int = 42):
    model = build_model(model_name, random_state=random_state)
    cv = TimeSeriesSplit(n_splits=5)

    if model_name == "logistic":
        param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    elif model_name == "random_forest":
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 8],
            "min_samples_leaf": [3, 5, 10],
        }
    elif model_name == "xgboost":
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_model(
    estimator,
    x_test,
    y_test,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_test)[:, 1]
    else:
        proba = estimator.predict(x_test)

    pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
        "decision_threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }
    return metrics, pred, proba


def find_best_threshold(y_true, probabilities) -> float:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.35, 0.66, 0.01):
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))

    return best_threshold


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    tune: bool = False,
    random_state: int = 42,
    optimize_threshold: bool = True,
) -> TrainResult:
    x_train, x_test, y_train, y_test = time_based_split(df, feature_columns)

    if tune:
        estimator, best_params = tune_model(model_name, x_train, y_train, random_state=random_state)
    else:
        estimator = build_model(model_name, random_state=random_state)
        best_params = None

    threshold = 0.5
    if optimize_threshold and len(x_train) >= 300:
        validation_start = int(len(x_train) * 0.8)
        x_fit, x_val = x_train.iloc[:validation_start], x_train.iloc[validation_start:]
        y_fit, y_val = y_train.iloc[:validation_start], y_train.iloc[validation_start:]

        threshold_estimator = build_model(model_name, random_state=random_state)
        threshold_estimator.fit(x_fit, y_fit)
        if hasattr(threshold_estimator, "predict_proba"):
            val_proba = threshold_estimator.predict_proba(x_val)[:, 1]
            threshold = find_best_threshold(y_val, val_proba)

    estimator.fit(x_train, y_train)
    metrics, pred, proba = evaluate_model(estimator, x_test, y_test, threshold=threshold)
    metrics["best_params"] = best_params

    predictions = df.iloc[-len(y_test) :][["Date", "Close", "target", "return_1d"]].copy()
    predictions["prediction"] = pred
    predictions["probability_up"] = proba

    importance = get_feature_importance(estimator, feature_columns)
    return TrainResult(model_name, estimator, metrics, predictions, importance)


def get_feature_importance(estimator, feature_columns: list[str]) -> pd.DataFrame | None:
    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["classifier"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
