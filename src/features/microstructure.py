"""
Microstructure Features Module

Computes market microstructure features for RL state vector:
- Trade sign estimation (Lee-Ready method)
- Order flow imbalance (OFI)
- Tick intensity (trades per unit time)
- Price impact measures
"""

import polars as pl
import numpy as np
from typing import List


def add_trade_sign(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estimate trade sign using tick rule (Lee-Ready method).

    +1 = uptick (buyer-initiated)
    -1 = downtick (seller-initiated)
    0 = zero tick (inherit previous)

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with trade_sign column added
    """
    # Price change
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("price_change")
    )

    # Initial sign based on price change
    df = df.with_columns(
        pl.when(pl.col("price_change") > 0)
        .then(pl.lit(1))
        .when(pl.col("price_change") < 0)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("trade_sign_raw")
    )

    # For zero ticks, propagate previous sign
    # Using forward fill
    df = df.with_columns(
        pl.when(pl.col("trade_sign_raw") != 0)
        .then(pl.col("trade_sign_raw"))
        .otherwise(None)
        .forward_fill()
        .fill_null(0)
        .alias("trade_sign")
    )

    df = df.drop("trade_sign_raw")

    return df


def add_order_flow_imbalance(
    df: pl.DataFrame,
    windows: List[int] = [20, 100],
) -> pl.DataFrame:
    """
    Add Order Flow Imbalance (OFI) features.

    OFI = sum(trade_sign * volume) over window
    Normalized OFI = OFI / total_volume

    Positive OFI = net buying pressure
    Negative OFI = net selling pressure

    Args:
        df: DataFrame with 'trade_sign' and 'volume' columns
        windows: Rolling window sizes

    Returns:
        DataFrame with OFI features added
    """
    # Ensure trade_sign exists
    if "trade_sign" not in df.columns:
        df = add_trade_sign(df)

    # Signed volume
    df = df.with_columns(
        (pl.col("trade_sign") * pl.col("volume")).alias("signed_volume")
    )

    for w in windows:
        # Rolling sum of signed volume
        df = df.with_columns(
            pl.col("signed_volume").rolling_sum(window_size=w).alias(f"ofi_{w}")
        )

        # Rolling sum of total volume
        df = df.with_columns(
            pl.col("volume").rolling_sum(window_size=w).alias(f"_total_vol_{w}")
        )

        # Normalized OFI
        df = df.with_columns(
            (pl.col(f"ofi_{w}") / (pl.col(f"_total_vol_{w}") + 1e-10))
            .alias(f"ofi_normalized_{w}")
        )

        df = df.drop(f"_total_vol_{w}")

    return df


def add_cumulative_trade_sign(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add cumulative trade sign (without volume weighting).

    Args:
        df: DataFrame with 'trade_sign' column
        windows: Rolling window sizes

    Returns:
        DataFrame with cumulative trade sign features added
    """
    if "trade_sign" not in df.columns:
        df = add_trade_sign(df)

    for w in windows:
        df = df.with_columns(
            pl.col("trade_sign").rolling_sum(window_size=w).alias(f"cum_sign_{w}")
        )

        # Normalize by window size
        df = df.with_columns(
            (pl.col(f"cum_sign_{w}") / w).alias(f"cum_sign_normalized_{w}")
        )

    return df


def add_tick_intensity(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add tick intensity features (for tick bars, this is based on duration).

    Intensity = number_of_ticks / time_duration
    Higher intensity = more active trading

    Args:
        df: DataFrame with 'duration_us' column (microseconds)
        windows: Rolling window sizes

    Returns:
        DataFrame with intensity features added
    """
    if "duration_us" not in df.columns:
        # If duration not available, skip
        return df

    # Ticks per millisecond (tick bars have 100 ticks each)
    df = df.with_columns(
        (100.0 / (pl.col("duration_us") / 1000 + 1e-10)).alias("tick_intensity")
    )

    for w in windows:
        df = df.with_columns(
            pl.col("tick_intensity").rolling_mean(window_size=w).alias(f"tick_intensity_ma_{w}")
        )

    return df


def add_price_impact(df: pl.DataFrame, windows: List[int] = [20, 100]) -> pl.DataFrame:
    """
    Add price impact measures (Kyle's Lambda approximation).

    Price impact = |price_change| / volume
    Higher impact = lower liquidity

    Args:
        df: DataFrame with 'close' and 'volume' columns
        windows: Rolling window sizes

    Returns:
        DataFrame with price impact features added
    """
    # Absolute return
    df = df.with_columns(
        ((pl.col("close") - pl.col("close").shift(1)).abs() / pl.col("close").shift(1))
        .alias("abs_return")
    )

    # Price impact per unit volume
    df = df.with_columns(
        (pl.col("abs_return") / (pl.col("volume") + 1e-10) * 1e6)  # Scale for readability
        .alias("price_impact")
    )

    for w in windows:
        df = df.with_columns(
            pl.col("price_impact").rolling_mean(window_size=w).alias(f"price_impact_ma_{w}")
        )

    return df


def add_amihud_illiquidity(df: pl.DataFrame, window: int = 100) -> pl.DataFrame:
    """
    Add Amihud illiquidity measure.

    ILLIQ = |return| / dollar_volume
    Higher ILLIQ = lower liquidity

    Args:
        df: DataFrame with 'close' and 'volume' columns
        window: Rolling window size

    Returns:
        DataFrame with Amihud measure added
    """
    # Dollar volume
    if "dollar_volume" not in df.columns:
        df = df.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume")
        )

    # Absolute return (if not present)
    if "abs_return" not in df.columns:
        df = df.with_columns(
            ((pl.col("close") - pl.col("close").shift(1)).abs() / pl.col("close").shift(1))
            .alias("abs_return")
        )

    # Amihud ILLIQ
    df = df.with_columns(
        (pl.col("abs_return") / (pl.col("dollar_volume") + 1e-10) * 1e9)  # Scale
        .alias("amihud_illiq")
    )

    # Rolling mean
    df = df.with_columns(
        pl.col("amihud_illiq").rolling_mean(window_size=window).alias("amihud_illiq_ma")
    )

    return df


def compute_all_microstructure_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all microstructure features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all microstructure features added
    """
    # Trade sign
    df = add_trade_sign(df)

    # Order flow imbalance
    df = add_order_flow_imbalance(df, windows=[20, 100])

    # Cumulative trade sign
    df = add_cumulative_trade_sign(df, windows=[20, 100])

    # Tick intensity
    df = add_tick_intensity(df, windows=[20, 100])

    # Price impact
    df = add_price_impact(df, windows=[20, 100])

    # Amihud illiquidity
    df = add_amihud_illiquidity(df, window=100)

    return df


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, "..")
    from data.bar_aggregator import load_bars_parquet

    bars = load_bars_parquet(
        "/Users/asefujiko/tools/tick_data/data/processed/toyota_7203_100tick_bars.parquet"
    )

    print("Computing microstructure features...")
    bars_with_features = compute_all_microstructure_features(bars)

    print(f"Columns: {bars_with_features.columns}")
    print(bars_with_features.head(5))
