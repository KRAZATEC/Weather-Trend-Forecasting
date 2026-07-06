"""
visualization.py
=================
Reusable, professionally-styled plotting functions for EDA (Phase 3),
model evaluation (Phase 6), and advanced analytics (Phase 7). Every
function saves its figure to outputs/figures/ and also returns the
Matplotlib/Plotly figure object for inline notebook display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 110,
})

PALETTE = "viridis"


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", out)
    return out


def plot_distribution(df: pd.DataFrame, col: str, name: str | None = None):
    """Histogram + KDE for a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col].dropna(), kde=True, color="#1C7293", ax=ax)
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    _save(fig, name or f"dist_{col}")
    return fig


def plot_boxplot_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                           name: str | None = None, top_n: int = 12):
    """Boxplot of value_col across the top-N most frequent categories of group_col."""
    top_groups = df[group_col].value_counts().head(top_n).index
    subset = df[df[group_col].isin(top_groups)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=subset, x=group_col, y=value_col, ax=ax, palette=PALETTE)
    ax.set_title(f"{value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, name or f"box_{value_col}_by_{group_col}")
    return fig


def plot_violin(df: pd.DataFrame, value_col: str, group_col: str, name: str | None = None, top_n: int = 8):
    top_groups = df[group_col].value_counts().head(top_n).index
    subset = df[df[group_col].isin(top_groups)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=subset, x=group_col, y=value_col, ax=ax, palette=PALETTE)
    ax.set_title(f"Violin Plot: {value_col.replace('_',' ').title()} by {group_col.replace('_',' ').title()}")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, name or f"violin_{value_col}_by_{group_col}")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None, name: str = "correlation_heatmap"):
    cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5), max(6, len(cols) * 0.45)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(cols) <= 15, fmt=".2f",
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix")
    _save(fig, name)
    return fig


def plot_pairplot(df: pd.DataFrame, cols: list[str], hue: str | None = None, name: str = "pairplot"):
    g = sns.pairplot(df[cols + ([hue] if hue else [])].dropna(), hue=hue, palette=PALETTE, corner=True)
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    out = config.FIGURES_DIR / f"{name}.png"
    g.savefig(out, bbox_inches="tight")
    plt.close(g.fig)
    logger.info("Saved figure -> %s", out)
    return g


def plot_time_trend(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "D",
                     agg: str = "mean", name: str | None = None):
    """Line chart of value_col aggregated over time."""
    ts = df.set_index(date_col)[value_col].resample(freq).agg(agg)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts.index, ts.values, color="#028090", linewidth=2)
    ax.fill_between(ts.index, ts.values, alpha=0.15, color="#028090")
    ax.set_title(f"{value_col.replace('_',' ').title()} Trend ({freq} {agg})")
    ax.set_ylabel(value_col.replace("_", " ").title())
    _save(fig, name or f"trend_{value_col}_{freq}")
    return fig


def plot_country_ranking(df: pd.DataFrame, value_col: str, group_col: str = "country",
                          top_n: int = 15, ascending: bool = False, name: str | None = None):
    ranking = df.groupby(group_col)[value_col].mean().sort_values(ascending=ascending).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=ranking.values, y=ranking.index, ax=ax, palette=PALETTE)
    direction = "Lowest" if ascending else "Highest"
    ax.set_title(f"{direction} Average {value_col.replace('_',' ').title()} by {group_col.title()}")
    ax.set_xlabel(value_col.replace("_", " ").title())
    _save(fig, name or f"ranking_{value_col}_by_{group_col}")
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None, name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=PALETTE, alpha=0.6, ax=ax)
    ax.set_title(f"{y.replace('_',' ').title()} vs {x.replace('_',' ').title()}")
    _save(fig, name or f"scatter_{x}_vs_{y}")
    return fig


def plot_seasonal_comparison(df: pd.DataFrame, value_col: str, season_col: str = "season",
                              name: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Winter", "Spring", "Summer", "Autumn"]
    sns.boxplot(data=df, x=season_col, y=value_col, order=order, palette="coolwarm", ax=ax)
    ax.set_title(f"{value_col.replace('_',' ').title()} by Season")
    _save(fig, name or f"season_{value_col}")
    return fig


def plot_prediction_vs_actual(y_true, y_pred, model_name: str, name: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#028090", s=15)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()
    _save(fig, name or f"pred_vs_actual_{model_name}")
    return fig


def plot_residuals(y_true, y_pred, model_name: str, name: str | None = None):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#B85042", s=15)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name}: Residuals vs Predicted")
    sns.histplot(residuals, kde=True, ax=axes[1], color="#B85042")
    axes[1].set_title("Residual Distribution")
    _save(fig, name or f"residuals_{model_name}")
    return fig


def plot_feature_importance(importances: pd.Series, model_name: str, top_n: int = 15, name: str | None = None):
    top = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette=PALETTE)
    ax.set_title(f"{model_name}: Top {top_n} Feature Importances")
    _save(fig, name or f"feature_importance_{model_name}")
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "RMSE", name: str = "model_comparison"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = results_df.sort_values(metric)
    sns.barplot(data=ordered, x=metric, y="Model", palette="crest", ax=ax)
    ax.set_title(f"Model Comparison by {metric}")
    _save(fig, name)
    return fig
