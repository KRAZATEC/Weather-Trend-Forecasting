"""
advanced_analytics.py
======================
Phase 7 deliverables: unsupervised anomaly detection (Isolation Forest,
Local Outlier Factor, DBSCAN), clustering (KMeans), dimensionality
reduction (PCA), feature-importance explainability (permutation importance
+ SHAP where available), and higher-level climate/weather-pattern analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def detect_anomalies_isolation_forest(df: pd.DataFrame, cols: list[str], contamination: float = 0.02):
    """Flag anomalous rows using Isolation Forest. Returns -1 (anomaly) / 1 (normal)."""
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    X = df[cols].fillna(df[cols].median())
    labels = model.fit_predict(X)
    logger.info("Isolation Forest flagged %d anomalies out of %d rows", (labels == -1).sum(), len(df))
    return labels, model


def detect_anomalies_lof(df: pd.DataFrame, cols: list[str], contamination: float = 0.02):
    """Flag anomalous rows using Local Outlier Factor."""
    X = df[cols].fillna(df[cols].median())
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
    labels = lof.fit_predict(X)
    logger.info("LOF flagged %d anomalies out of %d rows", (labels == -1).sum(), len(df))
    return labels


def cluster_dbscan(df: pd.DataFrame, cols: list[str], eps: float = 0.8, min_samples: int = 10):
    """Density-based clustering (also flags noise points as cluster -1)."""
    X = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("DBSCAN found %d clusters (+ noise points: %d)", n_clusters, (labels == -1).sum())
    return labels


def cluster_kmeans(df: pd.DataFrame, cols: list[str], n_clusters: int = 4):
    """KMeans clustering of weather regimes."""
    X = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    logger.info("KMeans formed %d clusters", n_clusters)
    return labels, model


def run_pca(df: pd.DataFrame, cols: list[str], n_components: int = 2):
    """Reduce dimensionality with PCA; returns (transformed array, fitted PCA)."""
    X = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()))
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)
    logger.info("PCA explained variance ratio: %s", np.round(pca.explained_variance_ratio_, 3))
    return transformed, pca


def compute_permutation_importance(model, X, y, n_repeats: int = 10) -> pd.Series:
    """Model-agnostic permutation feature importance."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 300):
    """Compute SHAP values for tree-based models (falls back gracefully if shap is unavailable)."""
    if not HAS_SHAP:
        logger.warning("shap not installed - skipping SHAP analysis")
        return None, None
    sample = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return shap_values, sample


def climate_trend_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """Linear-trend (slope per year) climate-change style analysis."""
    ts = df.set_index(date_col)[value_col].resample("D").mean().dropna()
    x = (ts.index - ts.index[0]).days.values
    slope, intercept = np.polyfit(x, ts.values, 1)
    annualized_change = slope * 365.25
    return {
        "slope_per_day": float(slope),
        "annualized_change_per_year": float(annualized_change),
        "start_value": float(ts.iloc[0]),
        "end_value": float(ts.iloc[-1]),
        "n_days": len(ts),
    }


def detect_heatwaves(df: pd.DataFrame, group_col: str = "location_name",
                      temp_col: str = "temperature_celsius",
                      date_col: str = "last_updated",
                      threshold_pct: float = 0.95, min_duration: int = 3) -> pd.DataFrame:
    """Flag heatwave events: temp above the location's 95th percentile for >= min_duration consecutive days."""
    events = []
    for loc, g in df.groupby(group_col):
        g = g.sort_values(date_col)
        threshold = g[temp_col].quantile(threshold_pct)
        is_hot = g[temp_col] >= threshold
        run_id = (is_hot != is_hot.shift()).cumsum()
        for _, run in g.groupby(run_id):
            if run[temp_col].iloc[0] >= threshold and len(run) >= min_duration:
                events.append({
                    "location": loc,
                    "start": run[date_col].iloc[0],
                    "end": run[date_col].iloc[-1],
                    "duration_days": len(run),
                    "max_temp": run[temp_col].max(),
                    "threshold": threshold,
                })
    return pd.DataFrame(events)


def country_ranking(df: pd.DataFrame, value_col: str, ascending: bool = False, top_n: int = 15) -> pd.Series:
    return df.groupby("country")[value_col].mean().sort_values(ascending=ascending).head(top_n)
