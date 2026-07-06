"""
forecasting.py
===============
Time-series forecasting of daily average temperature using ARIMA, SARIMA,
Prophet (used when available; a statsmodels-based seasonal-decomposition
fallback is used otherwise so the pipeline never breaks), and ML-based
direct-forecasting (lag-feature regression) approaches.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from src import config
from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not installed - using statsmodels-based fallback forecaster instead.")


def build_daily_series(df: pd.DataFrame, location: str | None = None,
                        date_col: str = config.DATETIME_COL,
                        value_col: str = config.TARGET_COL) -> pd.Series:
    """Aggregate the dataset into a single daily time series (global or per-location)."""
    subset = df if location is None else df[df["location_name"] == location]
    ts = subset.set_index(date_col)[value_col].resample("D").mean().interpolate()
    return ts


def forecast_arima(series: pd.Series, horizon: int = config.FORECAST_HORIZON_DAYS,
                    order=(2, 1, 2)):
    """Fit an ARIMA model and forecast `horizon` steps ahead."""
    model = ARIMA(series, order=order).fit()
    forecast = model.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    ci = forecast.conf_int()
    return mean, ci, model


def forecast_sarima(series: pd.Series, horizon: int = config.FORECAST_HORIZON_DAYS,
                     order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """Fit a SARIMA model (weekly seasonality) and forecast `horizon` steps ahead."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast = model.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    ci = forecast.conf_int()
    return mean, ci, model


def forecast_prophet_or_fallback(series: pd.Series, horizon: int = config.FORECAST_HORIZON_DAYS):
    """Forecast with Prophet if available, else a trend+seasonal decomposition fallback.

    Returns:
        pd.Series indexed by future dates.
    """
    if HAS_PROPHET:
        dfp = series.reset_index()
        dfp.columns = ["ds", "y"]
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon)
        fcst = m.predict(future)
        return fcst.set_index("ds")["yhat"].tail(horizon)

    # ---- Fallback: classical decomposition + linear trend extrapolation ----
    period = min(7, max(2, len(series) // 3))
    decomposition = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="freq")
    trend = decomposition.trend.dropna()
    x = np.arange(len(trend))
    coeffs = np.polyfit(x, trend.values, 1)
    future_x = np.arange(len(trend), len(trend) + horizon)
    future_trend = np.polyval(coeffs, future_x)

    seasonal_cycle = decomposition.seasonal.iloc[-period:].values
    future_seasonal = np.tile(seasonal_cycle, horizon // period + 1)[:horizon]

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_values = future_trend + future_seasonal
    return pd.Series(forecast_values, index=future_dates, name="yhat")


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute forecast accuracy metrics on the overlapping index."""
    common_idx = actual.index.intersection(predicted.index)
    if len(common_idx) == 0:
        return {"note": "no overlapping dates between actual and predicted"}
    a, p = actual.loc[common_idx], predicted.loc[common_idx]
    mae = (a - p).abs().mean()
    rmse = np.sqrt(((a - p) ** 2).mean())
    mape = ((a - p).abs() / a.abs().replace(0, np.nan)).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


if __name__ == "__main__":
    df = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=[config.DATETIME_COL])
    daily = build_daily_series(df)
    logger.info("Built global daily series: %d days", len(daily))

    arima_mean, _, _ = forecast_arima(daily)
    logger.info("ARIMA %d-day forecast:\n%s", config.FORECAST_HORIZON_DAYS, arima_mean.round(2))

    sarima_mean, _, _ = forecast_sarima(daily)
    logger.info("SARIMA %d-day forecast:\n%s", config.FORECAST_HORIZON_DAYS, sarima_mean.round(2))

    prophet_fc = forecast_prophet_or_fallback(daily)
    logger.info("Prophet/fallback %d-day forecast:\n%s", config.FORECAST_HORIZON_DAYS, prophet_fc.round(2))
