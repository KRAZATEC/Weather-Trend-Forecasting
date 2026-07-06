"""
config.py
=========
Central configuration for the Weather Trend Forecasting project.

All paths, constants, and hyperparameters used across the pipeline are
defined here so that every module (and every notebook) references a single
source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "GlobalWeatherRepository.csv"
CLEAN_DATA_FILE: Path = PROCESSED_DATA_DIR / "weather_clean.csv"
FEATURED_DATA_FILE: Path = PROCESSED_DATA_DIR / "weather_features.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Column groups (as found in the Kaggle "Global Weather Repository" dataset)
# ---------------------------------------------------------------------------
DATETIME_COL: str = "last_updated"

TARGET_COL: str = "temperature_celsius"

NUMERIC_COLS = [
    "latitude", "longitude",
    "temperature_celsius", "temperature_fahrenheit",
    "wind_mph", "wind_kph", "wind_degree",
    "pressure_mb", "pressure_in",
    "precip_mm", "precip_in",
    "humidity", "cloud",
    "feels_like_celsius", "feels_like_fahrenheit",
    "visibility_km", "visibility_miles",
    "uv_index", "gust_mph", "gust_kph",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5", "air_quality_PM10",
    "moon_illumination",
]

CATEGORICAL_COLS = [
    "country", "location_name", "timezone", "condition_text",
    "wind_direction", "moon_phase",
]

# ---------------------------------------------------------------------------
# Modelling configuration
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# Training controls
# Keep the default pipeline responsive on larger weather tables by capping
# the number of rows used by the most expensive estimators when necessary.
TRAINING_ROW_CAP: int = int(os.environ.get("WEATHER_TRAINING_ROW_CAP", "50000"))
BOOSTING_ROW_CAP: int = int(os.environ.get("WEATHER_BOOSTING_ROW_CAP", "25000"))
SVR_ROW_CAP: int = int(os.environ.get("WEATHER_SVR_ROW_CAP", "10000"))

MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 200, "max_depth": 12, "random_state": RANDOM_STATE, "n_jobs": -1},
    "GradientBoosting": {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.07,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
        "n_iter_no_change": 10,
        "validation_fraction": 0.1,
        "tol": 1e-4,
    },
    "XGBoost": {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.05, "random_state": RANDOM_STATE, "n_jobs": -1},
    "LightGBM": {"n_estimators": 250, "max_depth": -1, "learning_rate": 0.05, "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1},
    "LinearRegression": {},
    "DecisionTree": {"max_depth": 10, "random_state": RANDOM_STATE},
    "SVR": {"kernel": "rbf", "C": 10, "gamma": "scale"},
}

FORECAST_HORIZON_DAYS: int = 14

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_LEVEL: str = os.environ.get("WEATHER_LOG_LEVEL", "INFO")

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR,
             FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
