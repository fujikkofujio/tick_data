"""
Price-based Features Module

Computes price-related features for RL state vector:
- Returns (simple, log)
- Rolling statistics (mean, std, skew, kurtosis)
- Price position in range
- Volatility measures
"""

import polars as pl
import numpy as np
from typing import List


def add_returns(df: pl.DataFrame, periods: List[int] = [1, 5, 20, 100]) -> pl.DataFrame:
    """
    Add return features for multiple lookback periods.

    Args:
        df: DataFrame with 'close' column
        periods: List of lookback periods

    Returns:
        DataFrame with return columns added
    """
    for p in periods:
        # Simple return
        df = df.with_columns(
            ((pl.col("close") - pl.col("close").shift(p)) / pl.col("close").shift(p))
            .alias(f"return_{p}")
        )

        # Log return
        df = df.with_columns(
            (pl.col("close").log() - pl.col("close").shift(p).log())
            .alias(f"log_return_{p}")
        )

    return df


def add_rolling_stats(
    df: pl.DataFrame,
    column: str = "close",
    windows: List[int] = [20, 100],
) -> pl.DataFrame:
    """
    Add rolling statistics for a column.

    Args:
        df: Input DataFrame
        column: Column to compute statistics on
        windows: Rolling window sizes

    Returns:
        DataFrame with rolling statistics added
    """
    for w in windows:
        prefix = f"{column}_"

        # Rolling mean
        df = df.with_columns(
            pl.col(column).rolling_mean(window_size=w).alias(f"{prefix}ma_{w}")
        )

        # Rolling std (realized volatility for returns)
        df = df.with_columns(
            pl.col(column).rolling_std(window_size=w).alias(f"{prefix}std_{w}")
        )

        # Rolling min/max
        df = df.with_columns(
            [
                pl.col(column).rolling_min(window_size=w).alias(f"{prefix}min_{w}"),
                pl.col(column).rolling_max(window_size=w).alias(f"{prefix}max_{w}"),
            ]
        )

    return df


def add_price_position(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add price position within rolling range [0, 1].

    0 = at rolling low, 1 = at rolling high

    Args:
        df: DataFrame with 'close' column
        windows: Rolling window sizes

    Returns:
        DataFrame with position features added
    """
    for w in windows:
        high_col = f"close_max_{w}"
        low_col = f"close_min_{w}"

        # Ensure rolling min/max exist
        if high_col not in df.columns:
            df = df.with_columns(
                [
                    pl.col("close").rolling_min(window_size=w).alias(low_col),
                    pl.col("close").rolling_max(window_size=w).alias(high_col),
                ]
            )

        # Position in range
        df = df.with_columns(
            (
                (pl.col("close") - pl.col(low_col))
                / (pl.col(high_col) - pl.col(low_col) + 1e-10)
            ).alias(f"price_position_{w}")
        )

    return df


def add_realized_volatility(
    df: pl.DataFrame,
    return_col: str = "log_return_1",
    windows: List[int] = [20, 100],
) -> pl.DataFrame:
    """
    Add realized volatility measures.

    RV = sqrt(sum(r^2))

    Args:
        df: DataFrame with return column
        return_col: Name of the return column
        windows: Rolling window sizes

    Returns:
        DataFrame with volatility features added
    """
    for w in windows:
        # Squared returns
        df = df.with_columns(
            (pl.col(return_col) ** 2).alias("_sq_return")
        )

        # Rolling sum of squared returns
        df = df.with_columns(
            pl.col("_sq_return").rolling_sum(window_size=w).sqrt().alias(f"realized_vol_{w}")
        )

    # Clean up temp column
    df = df.drop("_sq_return")

    return df


def add_volatility_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add volatility ratio (short-term / long-term).

    High ratio indicates increasing volatility.

    Args:
        df: DataFrame with realized_vol_20 and realized_vol_100 columns

    Returns:
        DataFrame with vol_ratio added
    """
    if "realized_vol_20" in df.columns and "realized_vol_100" in df.columns:
        df = df.with_columns(
            (pl.col("realized_vol_20") / (pl.col("realized_vol_100") + 1e-10))
            .alias("vol_ratio")
        )

    return df


def compute_all_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all price-based features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all price features added
    """
    # Returns
    df = add_returns(df, periods=[1, 5, 20, 100])

    # Rolling statistics on close
    df = add_rolling_stats(df, column="close", windows=[20, 100])

    # Rolling statistics on returns
    df = add_rolling_stats(df, column="return_1", windows=[20, 100])

    # Price position
    df = add_price_position(df, windows=[20, 100])

    # Realized volatility
    df = add_realized_volatility(df, windows=[20, 100])

    # Volatility ratio
    df = add_volatility_ratio(df)

    return df


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, "..")
    from data.bar_aggregator import load_bars_parquet

    bars = load_bars_parquet(
        "/Users/asefujiko/tools/tick_data/data/processed/toyota_7203_100tick_bars.parquet"
    )

    print("Computing price features...")
    bars_with_features = compute_all_price_features(bars)

    print(f"Columns: {bars_with_features.columns}")
    print(bars_with_features.head(5))
