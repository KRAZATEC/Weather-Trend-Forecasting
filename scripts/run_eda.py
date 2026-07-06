"""
run_eda.py
==========
Generates the full EDA chart set (Phase 3 deliverable: 40+ professional
visualizations) into outputs/figures/.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, visualization as viz
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    df = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=[config.DATETIME_COL])
    df = pd.read_csv(config.FEATURED_DATA_FILE, parse_dates=[config.DATETIME_COL]) \
        if config.FEATURED_DATA_FILE.exists() else df

    n = 0

    # 1-10: Distributions
    for col in ["temperature_celsius", "humidity", "pressure_mb", "wind_kph", "cloud",
                "visibility_km", "uv_index", "precip_mm", "gust_kph", "moon_illumination"]:
        viz.plot_distribution(df, col); n += 1

    # 11-14: Boxplots by country
    for col in ["temperature_celsius", "humidity", "wind_kph", "pressure_mb"]:
        viz.plot_boxplot_by_group(df, col, "country"); n += 1

    # 15-16: Violin plots
    for col in ["temperature_celsius", "humidity"]:
        viz.plot_violin(df, col, "country"); n += 1

    # 17: Correlation heatmap
    numeric_cols = ["temperature_celsius", "humidity", "pressure_mb", "wind_kph", "cloud",
                     "visibility_km", "uv_index", "precip_mm", "gust_kph"]
    viz.plot_correlation_heatmap(df, numeric_cols); n += 1

    # 18: Pairplot
    viz.plot_pairplot(df, ["temperature_celsius", "humidity", "pressure_mb", "wind_kph"]); n += 1

    # 19-24: Time trends (daily) for key variables
    for col in ["temperature_celsius", "humidity", "pressure_mb", "wind_kph", "precip_mm", "uv_index"]:
        viz.plot_time_trend(df, config.DATETIME_COL, col, freq="D"); n += 1

    # 25-26: Weekly / Monthly aggregated trends
    viz.plot_time_trend(df, config.DATETIME_COL, "temperature_celsius", freq="W", name="trend_temp_weekly"); n += 1
    viz.plot_time_trend(df, config.DATETIME_COL, "temperature_celsius", freq="ME", name="trend_temp_monthly"); n += 1

    # 27-30: Country / continent style rankings
    viz.plot_country_ranking(df, "temperature_celsius", ascending=False, name="ranking_hottest_countries"); n += 1
    viz.plot_country_ranking(df, "temperature_celsius", ascending=True, name="ranking_coldest_countries"); n += 1
    viz.plot_country_ranking(df, "precip_mm", ascending=False, name="ranking_rainiest_countries"); n += 1
    viz.plot_country_ranking(df, "humidity", ascending=False, name="ranking_most_humid_countries"); n += 1

    # 31-34: Scatter relationships
    viz.plot_scatter(df, "humidity", "temperature_celsius"); n += 1
    viz.plot_scatter(df, "pressure_mb", "wind_kph"); n += 1
    viz.plot_scatter(df, "cloud", "uv_index"); n += 1
    viz.plot_scatter(df, "latitude", "temperature_celsius"); n += 1

    # 35: Seasonal comparison
    if "season" in df.columns:
        viz.plot_seasonal_comparison(df, "temperature_celsius"); n += 1
        viz.plot_seasonal_comparison(df, "humidity"); n += 1

    # 37-40: Additional distributions / categorical counts
    for col in ["air_quality_PM2.5", "air_quality_PM10", "wind_degree", "feels_like_celsius"]:
        viz.plot_distribution(df, col); n += 1

    # 41-42: Condition & moon-phase frequency
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    df["condition_text"].value_counts().plot(kind="barh", ax=ax, color="#1C7293")
    ax.set_title("Weather Condition Frequency")
    fig.savefig(config.FIGURES_DIR / "condition_frequency.png", bbox_inches="tight")
    plt.close(fig); n += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    df["moon_phase"].value_counts().plot(kind="bar", ax=ax, color="#6D2E46")
    ax.set_title("Moon Phase Frequency")
    fig.savefig(config.FIGURES_DIR / "moon_phase_frequency.png", bbox_inches="tight")
    plt.close(fig); n += 1

    # 43-44: Hourly / day-of-week pattern (using day_of_week if present)
    if "day_of_week" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        df.groupby("day_of_week")["temperature_celsius"].mean().plot(kind="bar", ax=ax, color="#028090")
        ax.set_title("Average Temperature by Day of Week")
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
        fig.savefig(config.FIGURES_DIR / "temp_by_day_of_week.png", bbox_inches="tight")
        plt.close(fig); n += 1

    if "month" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        df.groupby("month")["temperature_celsius"].mean().plot(kind="line", marker="o", ax=ax, color="#B85042")
        ax.set_title("Average Temperature by Month")
        fig.savefig(config.FIGURES_DIR / "temp_by_month.png", bbox_inches="tight")
        plt.close(fig); n += 1

    logger.info("EDA complete: %d charts generated in %s", n, config.FIGURES_DIR)
    print(f"Generated {n} charts -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
