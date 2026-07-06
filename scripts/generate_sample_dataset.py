"""
generate_sample_dataset.py
===========================
Generates a synthetic dataset that reproduces the exact column schema of the
Kaggle "Global Weather Repository" dataset
(https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository).

WHY THIS EXISTS
----------------
This project was built in a sandboxed environment with no network access to
kaggle.com, so the real CSV could not be downloaded. To keep every notebook,
script, model, and dashboard fully runnable end-to-end, this script produces
a realistic stand-in dataset: 60 countries, ~200 days of daily observations
each (~12,000 rows), with physically-plausible seasonal temperature curves
(by latitude/hemisphere), correlated meteorological fields, and injected
data-quality issues (duplicates, missing values, outliers) so that the
cleaning phase of the pipeline has real work to do.

TO USE THE REAL DATA INSTEAD
------------------------------
Download the CSV from Kaggle and drop it at:
    data/raw/GlobalWeatherRepository.csv
overwriting the synthetic file. Every downstream script/notebook reads from
that same path and column schema, so nothing else needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRIES = [
    ("United States", "New York", 40.71, -74.01, "America/New_York"),
    ("United States", "Los Angeles", 34.05, -118.24, "America/Los_Angeles"),
    ("United Kingdom", "London", 51.51, -0.13, "Europe/London"),
    ("France", "Paris", 48.85, 2.35, "Europe/Paris"),
    ("Germany", "Berlin", 52.52, 13.40, "Europe/Berlin"),
    ("India", "New Delhi", 28.61, 77.21, "Asia/Kolkata"),
    ("India", "Mumbai", 19.08, 72.88, "Asia/Kolkata"),
    ("China", "Beijing", 39.90, 116.40, "Asia/Shanghai"),
    ("China", "Shanghai", 31.23, 121.47, "Asia/Shanghai"),
    ("Japan", "Tokyo", 35.68, 139.65, "Asia/Tokyo"),
    ("Brazil", "Rio de Janeiro", -22.91, -43.17, "America/Sao_Paulo"),
    ("Brazil", "Sao Paulo", -23.55, -46.63, "America/Sao_Paulo"),
    ("Australia", "Sydney", -33.87, 151.21, "Australia/Sydney"),
    ("Australia", "Melbourne", -37.81, 144.96, "Australia/Melbourne"),
    ("Canada", "Toronto", 43.65, -79.38, "America/Toronto"),
    ("Canada", "Vancouver", 49.28, -123.12, "America/Vancouver"),
    ("Russia", "Moscow", 55.76, 37.62, "Europe/Moscow"),
    ("South Africa", "Cape Town", -33.92, 18.42, "Africa/Johannesburg"),
    ("South Africa", "Johannesburg", -26.20, 28.05, "Africa/Johannesburg"),
    ("Egypt", "Cairo", 30.04, 31.24, "Africa/Cairo"),
    ("Nigeria", "Lagos", 6.52, 3.38, "Africa/Lagos"),
    ("Kenya", "Nairobi", -1.29, 36.82, "Africa/Nairobi"),
    ("Mexico", "Mexico City", 19.43, -99.13, "America/Mexico_City"),
    ("Argentina", "Buenos Aires", -34.60, -58.38, "America/Argentina/Buenos_Aires"),
    ("Chile", "Santiago", -33.45, -70.67, "America/Santiago"),
    ("Spain", "Madrid", 40.42, -3.70, "Europe/Madrid"),
    ("Italy", "Rome", 41.90, 12.50, "Europe/Rome"),
    ("Netherlands", "Amsterdam", 52.37, 4.90, "Europe/Amsterdam"),
    ("Sweden", "Stockholm", 59.33, 18.07, "Europe/Stockholm"),
    ("Norway", "Oslo", 59.91, 10.75, "Europe/Oslo"),
    ("Finland", "Helsinki", 60.17, 24.94, "Europe/Helsinki"),
    ("Poland", "Warsaw", 52.23, 21.01, "Europe/Warsaw"),
    ("Turkey", "Istanbul", 41.01, 28.98, "Europe/Istanbul"),
    ("Saudi Arabia", "Riyadh", 24.71, 46.68, "Asia/Riyadh"),
    ("United Arab Emirates", "Dubai", 25.20, 55.27, "Asia/Dubai"),
    ("Indonesia", "Jakarta", -6.21, 106.85, "Asia/Jakarta"),
    ("Thailand", "Bangkok", 13.76, 100.50, "Asia/Bangkok"),
    ("Vietnam", "Hanoi", 21.03, 105.85, "Asia/Ho_Chi_Minh"),
    ("Philippines", "Manila", 14.60, 120.98, "Asia/Manila"),
    ("South Korea", "Seoul", 37.57, 126.98, "Asia/Seoul"),
    ("Pakistan", "Karachi", 24.86, 67.01, "Asia/Karachi"),
    ("Bangladesh", "Dhaka", 23.81, 90.41, "Asia/Dhaka"),
    ("New Zealand", "Auckland", -36.85, 174.76, "Pacific/Auckland"),
    ("Portugal", "Lisbon", 38.72, -9.14, "Europe/Lisbon"),
    ("Greece", "Athens", 37.98, 23.73, "Europe/Athens"),
    ("Ireland", "Dublin", 53.35, -6.26, "Europe/Dublin"),
    ("Switzerland", "Zurich", 47.38, 8.54, "Europe/Zurich"),
    ("Austria", "Vienna", 48.21, 16.37, "Europe/Vienna"),
    ("Belgium", "Brussels", 50.85, 4.35, "Europe/Brussels"),
    ("Denmark", "Copenhagen", 55.68, 12.57, "Europe/Copenhagen"),
    ("Iceland", "Reykjavik", 64.15, -21.94, "Atlantic/Reykjavik"),
    ("Morocco", "Casablanca", 33.57, -7.59, "Africa/Casablanca"),
    ("Peru", "Lima", -12.05, -77.04, "America/Lima"),
    ("Colombia", "Bogota", 4.71, -74.07, "America/Bogota"),
    ("Israel", "Tel Aviv", 32.08, 34.78, "Asia/Jerusalem"),
    ("Malaysia", "Kuala Lumpur", 3.14, 101.69, "Asia/Kuala_Lumpur"),
    ("Singapore", "Singapore", 1.35, 103.82, "Asia/Singapore"),
    ("Ukraine", "Kyiv", 50.45, 30.52, "Europe/Kyiv"),
    ("Kazakhstan", "Almaty", 43.24, 76.95, "Asia/Almaty"),
    ("Ethiopia", "Addis Ababa", 9.03, 38.74, "Africa/Addis_Ababa"),
]

CONDITIONS = [
    "Sunny", "Clear", "Partly cloudy", "Cloudy", "Overcast", "Mist",
    "Light rain", "Moderate rain", "Heavy rain", "Thunderstorm",
    "Light snow", "Snow", "Fog", "Patchy rain possible",
]

MOON_PHASES = [
    "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
    "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent",
]

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def seasonal_base_temp(latitude: float, day_of_year: int) -> float:
    """Physically-motivated seasonal temperature curve by latitude."""
    # Peak summer offset differs by hemisphere
    phase_shift = 172 if latitude >= 0 else 172 - 182  # ~June 21 vs ~Dec 21
    seasonal = 10 * np.cos(2 * np.pi * (day_of_year - phase_shift) / 365.25)
    lat_base = 30 - 0.55 * abs(latitude)  # hotter near equator
    return lat_base - seasonal * (0.4 + abs(latitude) / 120)


def generate(n_days: int = 200, start_date: str = "2024-06-01") -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []

    for country, city, lat, lon, tz in COUNTRIES:
        for d in dates:
            doy = d.timetuple().tm_yday
            base_temp = seasonal_base_temp(lat, doy)
            temp_c = base_temp + RNG.normal(0, 2.5)

            humidity = np.clip(70 - 0.5 * temp_c + RNG.normal(0, 10), 10, 100)
            pressure_mb = 1013 + RNG.normal(0, 6)
            wind_kph = np.clip(RNG.gamma(2, 6), 0, 90)
            cloud = np.clip(RNG.normal(45, 30), 0, 100)
            precip_mm = max(0, RNG.exponential(1.5) - 1.0) if RNG.random() < 0.35 else 0.0
            uv_index = np.clip((temp_c / 4) - 0.02 * abs(lat) + RNG.normal(0, 1.5), 0, 11)
            visibility_km = np.clip(RNG.normal(10, 3), 0.5, 20)
            feels_like_c = temp_c + (0.02 * humidity if temp_c > 20 else -0.03 * wind_kph)
            gust_kph = wind_kph * RNG.uniform(1.1, 1.6)

            condition = RNG.choice(
                CONDITIONS,
                p=_condition_probs(precip_mm, cloud, temp_c),
            )

            row = {
                "country": country,
                "location_name": city,
                "latitude": round(lat, 2),
                "longitude": round(lon, 2),
                "timezone": tz,
                "last_updated_epoch": int(d.timestamp()),
                "last_updated": d.strftime("%Y-%m-%d %H:%M"),
                "temperature_celsius": round(temp_c, 1),
                "temperature_fahrenheit": round(temp_c * 9 / 5 + 32, 1),
                "condition_text": condition,
                "wind_mph": round(wind_kph / 1.609, 1),
                "wind_kph": round(wind_kph, 1),
                "wind_degree": int(RNG.integers(0, 360)),
                "wind_direction": RNG.choice(WIND_DIRS),
                "pressure_mb": round(pressure_mb, 1),
                "pressure_in": round(pressure_mb * 0.02953, 2),
                "precip_mm": round(precip_mm, 1),
                "precip_in": round(precip_mm * 0.0394, 2),
                "humidity": int(humidity),
                "cloud": int(cloud),
                "feels_like_celsius": round(feels_like_c, 1),
                "feels_like_fahrenheit": round(feels_like_c * 9 / 5 + 32, 1),
                "visibility_km": round(visibility_km, 1),
                "visibility_miles": round(visibility_km / 1.609, 1),
                "uv_index": round(uv_index, 1),
                "gust_mph": round(gust_kph / 1.609, 1),
                "gust_kph": round(gust_kph, 1),
                "air_quality_Carbon_Monoxide": round(max(50, RNG.normal(300, 150)), 1),
                "air_quality_Ozone": round(max(0, RNG.normal(60, 25)), 1),
                "air_quality_Nitrogen_dioxide": round(max(0, RNG.normal(15, 10)), 2),
                "air_quality_Sulphur_dioxide": round(max(0, RNG.normal(8, 6)), 2),
                "air_quality_PM2.5": round(max(0, RNG.normal(20, 15)), 2),
                "air_quality_PM10": round(max(0, RNG.normal(30, 18)), 2),
                "air_quality_us-epa-index": int(RNG.integers(1, 6)),
                "air_quality_gb-defra-index": int(RNG.integers(1, 10)),
                "sunrise": "06:1%d AM" % RNG.integers(0, 9),
                "sunset": "07:%02d PM" % RNG.integers(0, 59),
                "moonrise": "%02d:%02d PM" % (RNG.integers(1, 11), RNG.integers(0, 59)),
                "moonset": "%02d:%02d AM" % (RNG.integers(1, 11), RNG.integers(0, 59)),
                "moon_phase": RNG.choice(MOON_PHASES),
                "moon_illumination": int(RNG.integers(0, 100)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Inject realistic data-quality issues for the cleaning phase -----
    df = _inject_missing_values(df, frac=0.02)
    df = _inject_outliers(df, frac=0.005)
    df = _inject_duplicates(df, n=40)

    return df


def _condition_probs(precip_mm: float, cloud: float, temp_c: float) -> np.ndarray:
    n = len(CONDITIONS)
    weights = np.ones(n)
    if precip_mm > 3:
        weights[CONDITIONS.index("Heavy rain")] += 6
        weights[CONDITIONS.index("Thunderstorm")] += 3
    elif precip_mm > 0.5:
        weights[CONDITIONS.index("Moderate rain")] += 5
        weights[CONDITIONS.index("Light rain")] += 5
    elif cloud > 70:
        weights[CONDITIONS.index("Overcast")] += 4
        weights[CONDITIONS.index("Cloudy")] += 4
    elif cloud < 20:
        weights[CONDITIONS.index("Sunny")] += 5
        weights[CONDITIONS.index("Clear")] += 4
    if temp_c < 0:
        weights[CONDITIONS.index("Light snow")] += 4
        weights[CONDITIONS.index("Snow")] += 3
    weights = weights / weights.sum()
    return weights


def _inject_missing_values(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    cols = ["humidity", "pressure_mb", "wind_kph", "uv_index", "visibility_km",
            "air_quality_PM2.5", "precip_mm"]
    n = len(df)
    for col in cols:
        idx = RNG.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    n = len(df)
    idx = RNG.choice(n, size=int(n * frac), replace=False)
    df.loc[idx, "temperature_celsius"] = df.loc[idx, "temperature_celsius"] * RNG.uniform(3, 5)
    idx2 = RNG.choice(n, size=int(n * frac), replace=False)
    df.loc[idx2, "wind_kph"] = df.loc[idx2, "wind_kph"] * RNG.uniform(6, 10)
    return df


def _inject_duplicates(df: pd.DataFrame, n: int) -> pd.DataFrame:
    dup_rows = df.sample(n=n, random_state=42)
    return pd.concat([df, dup_rows], ignore_index=True)


if __name__ == "__main__":
    from pathlib import Path

    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "GlobalWeatherRepository.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate()
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data):,} rows x {data.shape[1]} columns -> {out_path}")
