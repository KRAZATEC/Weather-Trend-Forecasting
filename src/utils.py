"""
utils.py
========
Shared utility functions: logging setup, timers, and small helpers reused
across the pipeline.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from src import config


def get_logger(name: str = "weather_pipeline") -> logging.Logger:
    """Create (or fetch) a configured logger that writes to console and file.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def timeit(func: Callable) -> Callable:
    """Decorator that logs the execution time of the wrapped function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s finished in %.2fs", func.__name__, elapsed)
        return result

    return wrapper


def reduce_memory_usage(df, verbose: bool = True):
    """Downcast numeric columns of a DataFrame to reduce memory footprint.

    Args:
        df: Input DataFrame.
        verbose: Whether to log the memory saved.

    Returns:
        The same DataFrame with downcast dtypes.
    """
    import numpy as np

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger = get_logger(__name__)
        logger.info(
            "Memory usage reduced from %.2f MB to %.2f MB (-%.1f%%)",
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem,
        )
    return df
