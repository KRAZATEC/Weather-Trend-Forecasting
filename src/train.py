"""
train.py
========
Trains multiple regression models to predict `temperature_celsius` from
engineered weather features, and persists each fitted model with joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def get_model_registry() -> dict:
    """Return a dict of {model_name: unfitted estimator} for every model to train."""
    registry = {
        "LinearRegression": LinearRegression(**config.MODEL_PARAMS["LinearRegression"]),
        "DecisionTree": DecisionTreeRegressor(**config.MODEL_PARAMS["DecisionTree"]),
        "RandomForest": RandomForestRegressor(**config.MODEL_PARAMS["RandomForest"]),
        "GradientBoosting": GradientBoostingRegressor(**config.MODEL_PARAMS["GradientBoosting"]),
        "SVR": SVR(**config.MODEL_PARAMS["SVR"]),
    }
    if HAS_XGBOOST:
        registry["XGBoost"] = XGBRegressor(**config.MODEL_PARAMS["XGBoost"], objective="reg:squarederror")
    if HAS_LIGHTGBM:
        registry["LightGBM"] = LGBMRegressor(**config.MODEL_PARAMS["LightGBM"])
    return registry


def _maybe_subsample(X_train: pd.DataFrame, y_train: pd.Series, max_rows: int):
    """Return a deterministic subsample when the training set is very large.

    This keeps the pipeline responsive on the full Kaggle-sized dataset while
    preserving full-data training for smaller runs.
    """
    if len(X_train) <= max_rows:
        return X_train, y_train

    sample_idx = X_train.sample(n=max_rows, random_state=config.RANDOM_STATE).index
    logger.info(
        "Using %d/%d rows for this model to keep training time manageable.",
        len(sample_idx),
        len(X_train),
    )
    return X_train.loc[sample_idx], y_train.loc[sample_idx]


def get_feature_matrix(df: pd.DataFrame, target_col: str = config.TARGET_COL):
    """Build the X, y matrices used for supervised training.

    Drops identifier / text / leakage columns, keeps numeric + engineered
    features only.
    """
    # NOTE ON LEAKAGE: heat_index, wind_chill, temp_feels_diff,
    # temperature_celsius_sq, feels_like_celsius/fahrenheit, and
    # humidity_temp_interaction are all deterministic functions of
    # *today's* temperature_celsius (the target). Including them would let
    # a model reconstruct the target almost exactly and inflate R^2 to ~1.0
    # without learning anything useful, so they are excluded here. Only the
    # lag/rolling versions of temperature (which use strictly past values)
    # are kept as legitimate autoregressive predictors.
    drop_cols = {
        target_col, "temperature_fahrenheit", "feels_like_fahrenheit", "feels_like_celsius",
        "temp_feels_diff", "heat_index", "wind_chill", "temperature_celsius_sq",
        "humidity_temp_interaction",
        config.DATETIME_COL, "last_updated_epoch", "country", "location_name",
        "timezone", "condition_text", "wind_direction", "moon_phase",
        "sunrise", "sunset", "moonrise", "moonset", "season",
    }
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
    X = numeric_df[feature_cols].fillna(numeric_df[feature_cols].median())
    y = df[target_col]
    return X, y, feature_cols


@timeit
def train_all_models(X_train, y_train) -> dict:
    """Fit every model in the registry on the training data.

    Returns:
        {model_name: fitted estimator}
    """
    fitted = {}
    for name, model in get_model_registry().items():
        logger.info("Training %s ...", name)
        if name == "SVR":
            fit_X, fit_y = _maybe_subsample(X_train, y_train, config.SVR_ROW_CAP)
        elif name in {"GradientBoosting", "XGBoost", "LightGBM"}:
            fit_X, fit_y = _maybe_subsample(X_train, y_train, config.BOOSTING_ROW_CAP)
        else:
            fit_X, fit_y = _maybe_subsample(X_train, y_train, config.TRAINING_ROW_CAP)

        model.fit(fit_X, fit_y)
        fitted[name] = model
        model_path = config.MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        logger.info("Saved %s -> %s", name, model_path)
    return fitted


def build_ensemble(fitted_models: dict, X, weights: dict | None = None) -> np.ndarray:
    """Simple weighted-average ensemble across all fitted models."""
    preds = np.column_stack([m.predict(X) for m in fitted_models.values()])
    if weights is None:
        return preds.mean(axis=1)
    w = np.array([weights.get(name, 1.0) for name in fitted_models])
    return (preds * w).sum(axis=1) / w.sum()


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURED_DATA_FILE, parse_dates=[config.DATETIME_COL])
    X, y, feature_cols = get_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    fitted = train_all_models(X_train, y_train)

    ensemble_pred = build_ensemble(fitted, X_test)
    joblib.dump({"feature_cols": feature_cols}, config.MODELS_DIR / "feature_cols.pkl")
    logger.info("Training complete. %d models saved to %s", len(fitted), config.MODELS_DIR)
