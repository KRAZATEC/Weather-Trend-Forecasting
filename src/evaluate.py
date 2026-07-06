"""
evaluate.py
===========
Computes regression metrics (MAE, MSE, RMSE, MAPE, R2), cross-validation
scores, and builds the model-comparison table (Phase 6 & 9 deliverables).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute the standard regression metric suite for one model's predictions.

    NOTE: temperature_celsius crosses zero, so ordinary MAPE explodes near
    0 degrees (division by a near-zero actual value) -- a well-known pitfall
    when applying MAPE to Celsius temperature. To keep MAPE meaningful we
    exclude points with |actual| < 2C from that one metric only; all other
    metrics use every point.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mape_mask = np.abs(y_true) >= 2.0
    if mape_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mape_mask], y_pred[mape_mask]) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}



def evaluate_models(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every fitted model on the held-out test set.

    Returns:
        DataFrame sorted by RMSE ascending (best model first).
    """
    rows = []
    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        metrics["Model"] = name
        rows.append(metrics)
    results = pd.DataFrame(rows)[["Model", "MAE", "MSE", "RMSE", "MAPE", "R2"]]
    results = results.sort_values("RMSE").reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))
    logger.info("Model comparison:\n%s", results.round(3).to_string(index=False))
    return results


def cross_validate_model(model, X, y, cv: int = config.CV_FOLDS, scoring: str = "neg_root_mean_squared_error"):
    """Run k-fold cross-validation and return fold scores + summary stats."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    if scoring.startswith("neg_"):
        scores = -scores
    return {"fold_scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}


def get_best_model(results: pd.DataFrame, fitted_models: dict):
    """Return (name, fitted model) for the top-ranked model in `results`."""
    best_name = results.iloc[0]["Model"]
    return best_name, fitted_models[best_name]
