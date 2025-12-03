"""
Volume-based Features Module

Computes volume-related features for RL state vector:
- Volume moving averages and ratios
- VWAP deviation
- Dollar volume
- Volume momentum
"""

import polars as pl
import numpy as np
from typing import List


def add_volume_ma(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add volume moving averages and ratios.

    Args:
        df: DataFrame with 'volume' column
        windows: Rolling window sizes

    Returns:
        DataFrame with volume MA features added
    """
    for w in windows:
        # Rolling mean volume
        df = df.with_columns(
            pl.col("volume").rolling_mean(window_size=w).alias(f"volume_ma_{w}")
        )

        # Volume ratio (current / MA)
        df = df.with_columns(
            (pl.col("volume") / (pl.col(f"volume_ma_{w}") + 1e-10))
            .alias(f"volume_ratio_{w}")
        )

    return df


def add_volume_zscore(df: pl.DataFrame, window: int = 100) -> pl.DataFrame:
    """
    Add volume z-score for detecting abnormal volume.

    Args:
        df: DataFrame with 'volume' column
        window: Rolling window size

    Returns:
        DataFrame with volume_zscore added
    """
    df = df.with_columns(
        [
            pl.col("volume").rolling_mean(window_size=window).alias("_vol_mean"),
            pl.col("volume").rolling_std(window_size=window).alias("_vol_std"),
        ]
    )

    df = df.with_columns(
        ((pl.col("volume") - pl.col("_vol_mean")) / (pl.col("_vol_std") + 1e-10))
        .alias("volume_zscore")
    )

    df = df.drop(["_vol_mean", "_vol_std"])

    return df


def add_vwap_deviation(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add VWAP deviation features.

    VWAP deviation = (price - VWAP) / VWAP
    Positive = price above VWAP (bullish), Negative = below (bearish)

    Args:
        df: DataFrame with 'close', 'volume' columns
        windows: Rolling window sizes

    Returns:
        DataFrame with VWAP deviation features added
    """
    # Calculate dollar volume
    if "dollar_volume" not in df.columns:
        df = df.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume")
        )

    for w in windows:
        # Rolling VWAP
        df = df.with_columns(
            [
                pl.col("dollar_volume").rolling_sum(window_size=w).alias(f"_dv_sum_{w}"),
                pl.col("volume").rolling_sum(window_size=w).alias(f"_vol_sum_{w}"),
            ]
        )

        df = df.with_columns(
            (pl.col(f"_dv_sum_{w}") / (pl.col(f"_vol_sum_{w}") + 1e-10))
            .alias(f"vwap_{w}")
        )

        # VWAP deviation
        df = df.with_columns(
            ((pl.col("close") - pl.col(f"vwap_{w}")) / (pl.col(f"vwap_{w}") + 1e-10))
            .alias(f"vwap_deviation_{w}")
        )

        df = df.drop([f"_dv_sum_{w}", f"_vol_sum_{w}"])

    return df


def add_dollar_volume_features(df: pl.DataFrame, window: int = 100) -> pl.DataFrame:
    """
    Add dollar volume features.

    Args:
        df: DataFrame with 'close', 'volume' columns
        window: Rolling window size

    Returns:
        DataFrame with dollar volume features added
    """
    # Dollar volume (if not already present)
    if "dollar_volume" not in df.columns:
        df = df.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume")
        )

    # Rolling mean dollar volume
    df = df.with_columns(
        pl.col("dollar_volume").rolling_mean(window_size=window).alias("dollar_volume_ma")
    )

    # Normalized dollar volume
    df = df.with_columns(
        (pl.col("dollar_volume") / (pl.col("dollar_volume_ma") + 1e-10))
        .alias("dollar_volume_normalized")
    )

    return df


def add_volume_momentum(df: pl.DataFrame, periods: List[int] = [5, 20]) -> pl.DataFrame:
    """
    Add volume momentum (rate of change).

    Args:
        df: DataFrame with 'volume' column
        periods: Lookback periods

    Returns:
        DataFrame with volume momentum features added
    """
    for p in periods:
        df = df.with_columns(
            ((pl.col("volume") - pl.col("volume").shift(p)) / (pl.col("volume").shift(p) + 1e-10))
            .alias(f"volume_momentum_{p}")
        )

    return df


def compute_all_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all volume-based features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all volume features added
    """
    # Volume moving averages and ratios
    df = add_volume_ma(df, windows=[20, 100])

    # Volume z-score
    df = add_volume_zscore(df, window=100)

    # VWAP deviation
    df = add_vwap_deviation(df, windows=[20, 100])

    # Dollar volume features
    df = add_dollar_volume_features(df, window=100)

    # Volume momentum
    df = add_volume_momentum(df, periods=[5, 20])

    return df


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, "..")
    from data.bar_aggregator import load_bars_parquet

    bars = load_bars_parquet(
        "/Users/asefujiko/tools/tick_data/data/processed/toyota_7203_100tick_bars.parquet"
    )

    print("Computing volume features...")
    bars_with_features = compute_all_volume_features(bars)

    print(f"Columns: {bars_with_features.columns}")
    print(bars_with_features.head(5))
