"""
feature_engineering.py
=======================
Derives new predictive features from the cleaned weather dataset: calendar
features, rolling statistics, lag features, comfort indices, and
interaction/polynomial features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)


@timeit
def add_calendar_features(df: pd.DataFrame, date_col: str = config.DATETIME_COL) -> pd.DataFrame:
    """Add year/month/day/week/quarter/weekend/season columns."""
    df = df.copy()
    dt = df[date_col]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["day_of_week"] = dt.dt.dayofweek
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = dt.dt.dayofyear

    def season_of(row):
        lat = row.get("latitude", 0)
        m = row["month"]
        north = m in (12, 1, 2)
        if lat >= 0:
            if m in (12, 1, 2):
                return "Winter"
            if m in (3, 4, 5):
                return "Spring"
            if m in (6, 7, 8):
                return "Summer"
            return "Autumn"
        else:
            if m in (12, 1, 2):
                return "Summer"
            if m in (3, 4, 5):
                return "Autumn"
            if m in (6, 7, 8):
                return "Winter"
            return "Spring"

    df["season"] = df.apply(season_of, axis=1)
    logger.info("Added calendar features")
    return df


@timeit
def add_comfort_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature-difference, heat index, and wind-chill features."""
    df = df.copy()
    if {"temperature_celsius", "feels_like_celsius"}.issubset(df.columns):
        df["temp_feels_diff"] = df["feels_like_celsius"] - df["temperature_celsius"]

    if {"temperature_celsius", "humidity"}.issubset(df.columns):
        t, rh = df["temperature_celsius"], df["humidity"]
        # Simplified Rothfusz heat index approximation (valid roughly for warm temps)
        df["heat_index"] = (
            -8.784695 + 1.61139411 * t + 2.338549 * rh - 0.14611605 * t * rh
            - 0.012308094 * t ** 2 - 0.016424828 * rh ** 2
            + 0.002211732 * t ** 2 * rh + 0.00072546 * t * rh ** 2
            - 0.000003582 * t ** 2 * rh ** 2
        )
        df.loc[t < 20, "heat_index"] = t[t < 20]  # heat index only meaningful when warm

    if {"temperature_celsius", "wind_kph"}.issubset(df.columns):
        t, v = df["temperature_celsius"], df["wind_kph"]
        wc = 13.12 + 0.6215 * t - 11.37 * v ** 0.16 + 0.3965 * t * v ** 0.16
        df["wind_chill"] = np.where((t <= 10) & (v > 4.8), wc, t)

    if {"humidity", "pressure_mb"}.issubset(df.columns):
        df["humidity_index"] = df["humidity"] / (df["pressure_mb"] / 1000)

    logger.info("Added comfort-index features (heat index, wind chill, humidity index)")
    return df


@timeit
def add_lag_rolling_features(df: pd.DataFrame, group_col: str = "location_name",
                              target_col: str = config.TARGET_COL,
                              lags: tuple[int, ...] = (1, 2, 3, 7),
                              windows: tuple[int, ...] = (3, 7, 14)) -> pd.DataFrame:
    """Add lag features and rolling mean/std, computed per location, sorted by date.

    Args:
        df: DataFrame containing group_col and a parsed datetime column.
        group_col: Column identifying independent time series (e.g. city).
        target_col: Column to lag / roll.
        lags: Lag periods (in rows / days) to generate.
        windows: Rolling window sizes (in rows / days).

    Returns:
        DataFrame with new lag_*, rollmean_*, rollstd_* columns.
    """
    df = df.sort_values([group_col, config.DATETIME_COL]).copy()
    grouped = df.groupby(group_col)[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = grouped.shift(lag)

    for window in windows:
        df[f"{target_col}_rollmean_{window}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"{target_col}_rollstd_{window}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())
        )

    # Fill early-series NaNs created by lag/rolling with the target's mean
    lag_roll_cols = [c for c in df.columns if c.startswith(f"{target_col}_lag_")
                      or c.startswith(f"{target_col}_rollmean_")
                      or c.startswith(f"{target_col}_rollstd_")]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(df[target_col].mean())

    logger.info("Added %d lag/rolling features", len(lag_roll_cols))
    return df.reset_index(drop=True)


@timeit
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a handful of domain-relevant interaction & polynomial features."""
    df = df.copy()
    if {"humidity", "temperature_celsius"}.issubset(df.columns):
        df["humidity_temp_interaction"] = df["humidity"] * df["temperature_celsius"]
    if {"wind_kph", "pressure_mb"}.issubset(df.columns):
        df["wind_pressure_interaction"] = df["wind_kph"] * df["pressure_mb"]
    if "temperature_celsius" in df.columns:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        arr = poly.fit_transform(df[["temperature_celsius"]])
        df["temperature_celsius_sq"] = arr[:, 1]
    logger.info("Added interaction and polynomial features")
    return df


@timeit
def select_features(df: pd.DataFrame, target_col: str = config.TARGET_COL,
                     top_k: int = 20) -> list[str]:
    """Select the top-k numeric features most correlated with the target.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Regression target.
        top_k: Number of features to keep.

    Returns:
        List of selected feature-column names.
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    correlations = numeric_df.corrwith(df[target_col]).abs().sort_values(ascending=False)
    selected = correlations.head(top_k).index.tolist()
    logger.info("Selected top %d features by |correlation| with %s", len(selected), target_col)
    return selected


@timeit
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline (Phase 4 deliverable)."""
    df = add_calendar_features(df)
    df = add_comfort_indices(df)
    df = add_lag_rolling_features(df)
    df = add_interaction_features(df)
    return df


if __name__ == "__main__":
    clean = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=[config.DATETIME_COL])
    featured = build_features(clean)
    featured.to_csv(config.FEATURED_DATA_FILE, index=False)
    logger.info("Saved feature-engineered dataset to %s (%d cols)",
                config.FEATURED_DATA_FILE, featured.shape[1])
