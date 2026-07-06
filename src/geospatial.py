"""
geospatial.py
==============
Phase 8 deliverables: interactive Folium maps of global weather
(temperature, humidity, rainfall, air quality, wind), plus a heatmap-style
weather-density view.
"""

from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import HeatMap

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def _location_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the dataset to one latest row per location for map plotting."""
    return (
        df.sort_values(config.DATETIME_COL)
        .groupby("location_name")
        .tail(1)
        .reset_index(drop=True)
    )


def build_marker_map(df: pd.DataFrame, value_col: str, title: str, colormap: str = "RdYlBu_r") -> folium.Map:
    """Interactive marker map colored by `value_col`."""
    snap = _location_snapshot(df)
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    vmin, vmax = snap[value_col].min(), snap[value_col].max()

    def color_for(v):
        pct = 0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        # simple blue -> red gradient
        r = int(255 * pct)
        b = int(255 * (1 - pct))
        return f"#{r:02x}64{b:02x}"

    for _, row in snap.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=folium.Popup(
                f"<b>{row['location_name']}, {row['country']}</b><br>"
                f"{value_col.replace('_', ' ').title()}: {row[value_col]:.1f}",
                max_width=250,
            ),
            color=color_for(row[value_col]),
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    title_html = f'<h3 style="text-align:center;font-family:sans-serif">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def build_heatmap(df: pd.DataFrame, value_col: str, title: str) -> folium.Map:
    """Density-weighted heatmap of `value_col` across all locations."""
    snap = _location_snapshot(df)
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    heat_data = snap[["latitude", "longitude", value_col]].dropna().values.tolist()
    HeatMap(heat_data, radius=25, blur=15, max_zoom=4).add_to(m)
    title_html = f'<h3 style="text-align:center;color:white;font-family:sans-serif">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def generate_all_maps(df: pd.DataFrame) -> dict:
    """Generate the full Phase-8 map set and save as standalone HTML files."""
    maps = {
        "temperature_map": build_marker_map(df, "temperature_celsius", "Global Temperature Map (\u00b0C)"),
        "humidity_map": build_marker_map(df, "humidity", "Global Humidity Map (%)"),
        "rainfall_map": build_marker_map(df, "precip_mm", "Global Rainfall Map (mm)"),
        "wind_map": build_marker_map(df, "wind_kph", "Global Wind Speed Map (kph)"),
        "air_quality_map": build_marker_map(df, "air_quality_PM2.5", "Global Air Quality Map (PM2.5)"),
        "temperature_density_heatmap": build_heatmap(df, "temperature_celsius", "Weather Density: Temperature"),
    }
    for name, m in maps.items():
        out_path = config.OUTPUTS_DIR / "figures" / f"{name}.html"
        m.save(str(out_path))
        logger.info("Saved interactive map -> %s", out_path)
    return maps


if __name__ == "__main__":
    data = pd.read_csv(config.CLEAN_DATA_FILE, parse_dates=[config.DATETIME_COL])
    generate_all_maps(data)
