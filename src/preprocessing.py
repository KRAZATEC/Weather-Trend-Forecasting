"""
preprocessing.py
=================
Data cleaning pipeline: duplicate removal, missing-value imputation, outlier
handling (IQR & Z-score), datetime parsing, categorical encoding, and
scaling. Implemented as small, composable, reusable functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)


@timeit
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows (%d -> %d)", before - len(df), before, len(df))
    return df


@timeit
def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values.

    Numeric columns: median (default) or mean imputation.
    Categorical columns: mode imputation.

    Args:
        df: Input DataFrame.
        strategy: 'median' or 'mean' for numeric columns.

    Returns:
        DataFrame with no missing values in known numeric/categorical cols.
    """
    df = df.copy()
    numeric_cols = [c for c in config.NUMERIC_COLS if c in df.columns]
    categorical_cols = [c for c in config.CATEGORICAL_COLS if c in df.columns]

    for col in numeric_cols:
        if df[col].isna().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)

    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    remaining = df.isna().sum().sum()
    logger.info("Missing-value imputation complete. Remaining NaNs: %d", remaining)
    return df


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return a boolean mask flagging IQR-based outliers."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return a boolean mask flagging Z-score-based outliers."""
    z = (series - series.mean()) / series.std(ddof=0)
    return z.abs() > threshold


@timeit
def handle_outliers(df: pd.DataFrame, cols: list[str] | None = None,
                     method: str = "iqr", action: str = "clip") -> pd.DataFrame:
    """Detect and treat outliers in numeric columns.

    Args:
        df: Input DataFrame.
        cols: Columns to check (defaults to config.NUMERIC_COLS present in df).
        method: 'iqr' or 'zscore'.
        action: 'clip' (winsorize to bounds) or 'remove' (drop rows).

    Returns:
        Outlier-treated DataFrame.
    """
    df = df.copy()
    cols = cols or [c for c in config.NUMERIC_COLS if c in df.columns]
    total_flagged = 0

    for col in cols:
        if method == "iqr":
            mask = detect_outliers_iqr(df[col])
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            mask = detect_outliers_zscore(df[col])
            lower, upper = df[col].mean() - 3 * df[col].std(), df[col].mean() + 3 * df[col].std()

        total_flagged += int(mask.sum())
        if action == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "remove":
            df = df.loc[~mask]

    logger.info("Outlier handling (%s/%s): %d values flagged", method, action, total_flagged)
    return df.reset_index(drop=True)


@timeit
def parse_datetime(df: pd.DataFrame, col: str = config.DATETIME_COL) -> pd.DataFrame:
    """Convert the datetime column to pandas datetime dtype."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    n_bad = df[col].isna().sum()
    if n_bad:
        logger.warning("%d rows had unparseable datetimes and were dropped", n_bad)
        df = df.dropna(subset=[col])
    return df.reset_index(drop=True)


@timeit
def encode_categoricals(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns, returning the fitted encoders.

    Returns:
        (encoded_df, {column: fitted LabelEncoder})
    """
    df = df.copy()
    cols = cols or [c for c in config.CATEGORICAL_COLS if c in df.columns]
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    logger.info("Encoded %d categorical columns", len(cols))
    return df, encoders


@timeit
def scale_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric columns (zero mean, unit variance)."""
    df = df.copy()
    scaler = StandardScaler()
    df[[f"{c}_scaled" for c in cols]] = scaler.fit_transform(df[cols])
    return df, scaler


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run basic data-validation checks and return a report dict."""
    report = {
        "n_rows": len(df),
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_total": int(df.isna().sum().sum()),
        "negative_humidity": int((df.get("humidity", pd.Series(dtype=float)) < 0).sum()),
        "humidity_over_100": int((df.get("humidity", pd.Series(dtype=float)) > 100).sum()),
        "invalid_latitude": int((df.get("latitude", pd.Series(dtype=float)).abs() > 90).sum()),
        "invalid_longitude": int((df.get("longitude", pd.Series(dtype=float)).abs() > 180).sum()),
    }
    logger.info("Validation report: %s", report)
    return report


@timeit
def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end (Phase 2 deliverable)."""
    df = remove_duplicates(df)
    df = parse_datetime(df)
    df = handle_missing_values(df, strategy="median")
    df = handle_outliers(df, method="iqr", action="clip")
    validate_dataset(df)
    return df


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw = load_raw_data()
    clean = clean_pipeline(raw)
    clean.to_csv(config.CLEAN_DATA_FILE, index=False)
    logger.info("Saved cleaned dataset to %s", config.CLEAN_DATA_FILE)
