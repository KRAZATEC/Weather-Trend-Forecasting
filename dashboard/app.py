"""
app.py
======
Streamlit dashboard for the Weather Trend Forecasting project.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

st.set_page_config(
    page_title="Weather Trend Forecasting",
    page_icon="\U0001F324\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / dark-mode toggle
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


@st.cache_data
def load_data() -> pd.DataFrame:
    path = config.FEATURED_DATA_FILE if config.FEATURED_DATA_FILE.exists() else config.CLEAN_DATA_FILE
    df = pd.read_csv(path, parse_dates=[config.DATETIME_COL])
    return df


@st.cache_resource
def load_model_comparison() -> pd.DataFrame | None:
    path = config.REPORTS_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_models() -> dict:
    models = {}
    for name in ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "LinearRegression"]:
        p = config.MODELS_DIR / f"{name}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
    return models


df = load_data()
comparison = load_model_comparison()
models = load_models()

# ---------------------------------------------------------------------------
# Sidebar navigation & filters
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F324\uFE0F Weather Trend Forecasting")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset", "EDA", "Climate Trends", "Forecast", "Model Comparison", "Maps", "About"],
)

st.session_state.dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.dark_mode)
template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
countries = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect("Country", countries, default=countries[:8])
date_min, date_max = df[config.DATETIME_COL].min(), df[config.DATETIME_COL].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max))

filtered = df[df["country"].isin(selected_countries)] if selected_countries else df
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered[config.DATETIME_COL] >= pd.Timestamp(date_range[0]))
        & (filtered[config.DATETIME_COL] <= pd.Timestamp(date_range[1]))
    ]

st.sidebar.markdown("---")
st.sidebar.download_button(
    "\u2b07\ufe0f Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_weather_data.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Weather Trend Forecasting Dashboard")
    st.caption("Machine learning & advanced analytics on the Global Weather Repository dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Countries", df["country"].nunique())
    c3.metric("Avg Temperature", f"{df['temperature_celsius'].mean():.1f} \u00b0C")
    c4.metric("Date Range", f"{(date_max - date_min).days} days")

    st.markdown("### Global Temperature Snapshot")
    latest = df.sort_values(config.DATETIME_COL).groupby("location_name").tail(1)
    fig = px.scatter_geo(
        latest, lat="latitude", lon="longitude", color="temperature_celsius",
        hover_name="location_name", hover_data={"country": True, "temperature_celsius": ":.1f"},
        color_continuous_scale="RdYlBu_r", projection="natural earth", template=template,
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use the sidebar to filter by country and date range, then explore the "
        "EDA, Climate Trends, Forecast, Model Comparison, and Maps pages."
    )

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
elif page == "Dataset":
    st.title("Dataset Overview")
    st.write(f"**Shape:** {filtered.shape[0]:,} rows x {filtered.shape[1]} columns")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(filtered.describe().T, use_container_width=True)

    st.markdown("### Missing Values")
    missing = filtered.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.bar_chart(missing)
    else:
        st.success("No missing values in the filtered data.")

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        var = st.selectbox("Variable to explore", numeric_cols, index=numeric_cols.index("temperature_celsius")
                            if "temperature_celsius" in numeric_cols else 0)
        fig = px.histogram(filtered, x=var, nbins=40, marginal="box", template=template,
                            color_discrete_sequence=["#028090"])
        fig.update_layout(title=f"Distribution of {var}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.box(filtered, x="country", y=var, template=template, color="country")
        fig2.update_layout(showlegend=False, title=f"{var} by Country")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Correlation Matrix")
    corr_cols = [c for c in ["temperature_celsius", "humidity", "pressure_mb", "wind_kph",
                              "cloud", "visibility_km", "uv_index", "precip_mm"] if c in filtered.columns]
    corr = filtered[corr_cols].corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", template=template)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# CLIMATE TRENDS
# ---------------------------------------------------------------------------
elif page == "Climate Trends":
    st.title("Climate Trends")
    trend_var = st.selectbox("Metric", ["temperature_celsius", "humidity", "precip_mm", "uv_index"])
    freq_label = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[freq_label]

    ts = filtered.set_index(config.DATETIME_COL)[trend_var].resample(freq).mean()
    fig = px.line(ts, template=template, labels={"value": trend_var, config.DATETIME_COL: "Date"})
    fig.update_traces(line_color="#028090")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Seasonal Comparison")
    if "season" in filtered.columns:
        fig2 = px.box(filtered, x="season", y=trend_var, category_orders={"season": ["Winter", "Spring", "Summer", "Autumn"]},
                       template=template, color="season")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------
elif page == "Forecast":
    st.title("Temperature Forecast")
    location = st.selectbox("Location", sorted(df["location_name"].unique()))
    horizon = st.slider("Forecast horizon (days)", 3, 30, 14)

    if st.button("Run Forecast"):
        from src.forecasting import build_daily_series, forecast_sarima

        series = build_daily_series(df, location=location)
        with st.spinner("Fitting SARIMA model..."):
            mean, ci, _ = forecast_sarima(series, horizon=horizon)

        hist_df = series.reset_index()
        hist_df.columns = ["date", "value"]
        fut_df = mean.reset_index()
        fut_df.columns = ["date", "value"]

        fig = px.line(template=template)
        fig.add_scatter(x=hist_df["date"], y=hist_df["value"], name="Historical", line=dict(color="#028090"))
        fig.add_scatter(x=fut_df["date"], y=fut_df["value"], name="Forecast", line=dict(color="#B85042", dash="dash"))
        fig.update_layout(title=f"{horizon}-Day Temperature Forecast: {location}", yaxis_title="Temperature (\u00b0C)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fut_df.rename(columns={"value": "forecast_temp_c"}))

    st.markdown("---")
    st.markdown("### Manual Prediction Input")
    st.caption("Predict temperature from current weather conditions using the trained ML model")
    if models:
        model_name = st.selectbox("Model", list(models.keys()))
        c1, c2, c3 = st.columns(3)
        humidity_in = c1.slider("Humidity (%)", 0, 100, 60)
        pressure_in = c2.slider("Pressure (mb)", 970, 1050, 1013)
        wind_in = c3.slider("Wind Speed (kph)", 0, 100, 15)
        c4, c5, c6 = st.columns(3)
        cloud_in = c4.slider("Cloud Cover (%)", 0, 100, 40)
        uv_in = c5.slider("UV Index", 0, 11, 5)
        lat_in = c6.slider("Latitude", -60, 70, 20)

        st.caption(
            "This is a simplified demo input covering a subset of model features; "
            "remaining features are filled with dataset medians."
        )
        if st.button("Predict Temperature"):
            fc_info = joblib.load(config.MODELS_DIR / "feature_cols.pkl")
            feature_cols = fc_info["feature_cols"]
            base = df[feature_cols].median()
            base["humidity"] = humidity_in
            base["pressure_mb"] = pressure_in
            base["wind_kph"] = wind_in
            base["cloud"] = cloud_in
            base["uv_index"] = uv_in
            base["latitude"] = lat_in
            X_input = pd.DataFrame([base])[feature_cols]
            pred = models[model_name].predict(X_input)[0]
            st.success(f"Predicted temperature: **{pred:.1f} \u00b0C** ({model_name})")
    else:
        st.warning("No trained models found. Run `python -m src.train` first.")

# ---------------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)
        fig = px.bar(comparison.sort_values("RMSE"), x="RMSE", y="Model", orientation="h",
                     template=template, color="RMSE", color_continuous_scale="Blues_r")
        st.plotly_chart(fig, use_container_width=True)

        best = comparison.sort_values("RMSE").iloc[0]
        st.success(f"\U0001F3C6 Best model: **{best['Model']}** (RMSE = {best['RMSE']:.3f}, R\u00b2 = {best['R2']:.3f})")
    else:
        st.warning("Run `python scripts/run_model_evaluation.py` to generate the comparison table.")

# ---------------------------------------------------------------------------
# MAPS
# ---------------------------------------------------------------------------
elif page == "Maps":
    st.title("Geospatial Weather Maps")
    map_var = st.selectbox(
        "Variable",
        ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "air_quality_PM2.5"],
    )
    latest = df.sort_values(config.DATETIME_COL).groupby("location_name").tail(1)
    fig = px.scatter_geo(
        latest, lat="latitude", lon="longitude", color=map_var, size=map_var if (latest[map_var] >= 0).all() else None,
        hover_name="location_name", hover_data={"country": True}, projection="natural earth",
        color_continuous_scale="RdYlBu_r", template=template,
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("About This Project")
    st.markdown(
        """
        **Weather Trend Forecasting** is an end-to-end data science project covering:

        - Data cleaning & validation
        - 40+ exploratory visualizations
        - Feature engineering (lag/rolling/comfort indices)
        - Multiple ML models (Linear, Tree, Ensemble, Boosting) + time-series forecasting (ARIMA/SARIMA/Prophet)
        - Advanced analytics: clustering, anomaly detection, PCA, SHAP explainability
        - Geospatial analysis with interactive maps
        - This Streamlit dashboard

        Built with Python, Pandas, Scikit-learn, XGBoost, LightGBM, Statsmodels, Plotly, and Streamlit.

        See the project `README.md` for full documentation.
        """
    )
