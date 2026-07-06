"""
data_loader.py
===============
Loading and initial profiling of the Global Weather Repository dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)


@timeit
def load_raw_data(path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw weather CSV into a DataFrame.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Raw (unmodified) DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. Run "
            "`python scripts/generate_sample_dataset.py` or place the Kaggle "
            "CSV at this location."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows x %s columns", *df.shape)
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compute a standard profiling summary of a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with shape, dtypes, memory usage, duplicate count, and
        missing-value counts.
    """
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "numeric_summary": df.describe(include="number").to_dict(),
    }
    return profile


def print_profile(df: pd.DataFrame) -> None:
    """Pretty-print a dataset profile to stdout (Phase 1 deliverable)."""
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Shape               : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage        : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate rows      : {df.duplicated().sum():,}")
    print(f"Total missing cells : {df.isna().sum().sum():,}")
    print("\nColumn dtypes:")
    print(df.dtypes.value_counts())
    print("\nTop 10 columns by missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nNumeric summary statistics:")
    print(df.describe(include="number").T.round(2))


if __name__ == "__main__":
    data = load_raw_data()
    print_profile(data)
