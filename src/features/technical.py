"""
Technical Indicators Module

Computes technical analysis indicators for RL state vector:
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Time-based features
"""

import polars as pl
import numpy as np
from typing import List, Tuple


def add_rsi(df: pl.DataFrame, periods: List[int] = [14, 20]) -> pl.DataFrame:
    """
    Add RSI (Relative Strength Index) indicator.

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss

    RSI > 70 = overbought
    RSI < 30 = oversold

    Args:
        df: DataFrame with 'close' column
        periods: RSI calculation periods

    Returns:
        DataFrame with RSI columns added
    """
    # Price change
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("_price_change")
    )

    # Gains and losses
    df = df.with_columns(
        [
            pl.when(pl.col("_price_change") > 0)
            .then(pl.col("_price_change"))
            .otherwise(0.0)
            .alias("_gain"),
            pl.when(pl.col("_price_change") < 0)
            .then(-pl.col("_price_change"))
            .otherwise(0.0)
            .alias("_loss"),
        ]
    )

    for p in periods:
        # Exponential moving average of gains/losses
        df = df.with_columns(
            [
                pl.col("_gain").ewm_mean(span=p, adjust=False).alias(f"_avg_gain_{p}"),
                pl.col("_loss").ewm_mean(span=p, adjust=False).alias(f"_avg_loss_{p}"),
            ]
        )

        # RS and RSI
        df = df.with_columns(
            (pl.col(f"_avg_gain_{p}") / (pl.col(f"_avg_loss_{p}") + 1e-10)).alias(f"_rs_{p}")
        )

        df = df.with_columns(
            (100.0 - 100.0 / (1.0 + pl.col(f"_rs_{p}"))).alias(f"rsi_{p}")
        )

        # Normalize RSI to [-1, 1] for RL
        df = df.with_columns(
            ((pl.col(f"rsi_{p}") - 50.0) / 50.0).alias(f"rsi_{p}_normalized")
        )

        # Clean up
        df = df.drop([f"_avg_gain_{p}", f"_avg_loss_{p}", f"_rs_{p}"])

    df = df.drop(["_price_change", "_gain", "_loss"])

    return df


def add_bollinger_bands(
    df: pl.DataFrame,
    periods: List[int] = [20],
    num_std: float = 2.0,
) -> pl.DataFrame:
    """
    Add Bollinger Bands indicator.

    BB_upper = MA + k * std
    BB_lower = MA - k * std
    BB_position = (price - lower) / (upper - lower)  # [0, 1]
    BB_width = (upper - lower) / MA

    Args:
        df: DataFrame with 'close' column
        periods: Moving average periods
        num_std: Number of standard deviations for bands

    Returns:
        DataFrame with BB columns added
    """
    for p in periods:
        # Moving average and std
        df = df.with_columns(
            [
                pl.col("close").rolling_mean(window_size=p).alias(f"bb_ma_{p}"),
                pl.col("close").rolling_std(window_size=p).alias(f"bb_std_{p}"),
            ]
        )

        # Upper and lower bands
        df = df.with_columns(
            [
                (pl.col(f"bb_ma_{p}") + num_std * pl.col(f"bb_std_{p}")).alias(f"bb_upper_{p}"),
                (pl.col(f"bb_ma_{p}") - num_std * pl.col(f"bb_std_{p}")).alias(f"bb_lower_{p}"),
            ]
        )

        # Position within bands [0, 1]
        df = df.with_columns(
            (
                (pl.col("close") - pl.col(f"bb_lower_{p}"))
                / (pl.col(f"bb_upper_{p}") - pl.col(f"bb_lower_{p}") + 1e-10)
            ).alias(f"bb_position_{p}")
        )

        # Band width (relative to MA)
        df = df.with_columns(
            (
                (pl.col(f"bb_upper_{p}") - pl.col(f"bb_lower_{p}"))
                / (pl.col(f"bb_ma_{p}") + 1e-10)
            ).alias(f"bb_width_{p}")
        )

    return df


def add_macd(
    df: pl.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence) indicator.

    MACD line = EMA_fast - EMA_slow
    Signal line = EMA of MACD line
    Histogram = MACD line - Signal line

    Args:
        df: DataFrame with 'close' column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        DataFrame with MACD columns added
    """
    # EMAs
    df = df.with_columns(
        [
            pl.col("close").ewm_mean(span=fast_period, adjust=False).alias("_ema_fast"),
            pl.col("close").ewm_mean(span=slow_period, adjust=False).alias("_ema_slow"),
        ]
    )

    # MACD line
    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
    )

    # Signal line
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=signal_period, adjust=False).alias("macd_signal")
    )

    # Histogram
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
    )

    # Normalize by price for comparability
    df = df.with_columns(
        [
            (pl.col("macd_line") / (pl.col("close") + 1e-10) * 100).alias("macd_line_normalized"),
            (pl.col("macd_histogram") / (pl.col("close") + 1e-10) * 100).alias("macd_histogram_normalized"),
        ]
    )

    df = df.drop(["_ema_fast", "_ema_slow"])

    return df


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add time-based features for intraday patterns.

    - Time cyclical encoding (sin/cos)
    - Session flags (opening, lunch, closing)

    Args:
        df: DataFrame with 'time_start_us' column

    Returns:
        DataFrame with time features added
    """
    if "time_start_us" not in df.columns:
        return df

    # Trading hours: 9:00-11:30 (AM), 12:30-15:00 (PM)
    # Total trading minutes: 150 + 150 = 300 minutes
    TRADING_START_US = 9 * 3600 * 1_000_000  # 9:00 in microseconds
    TRADING_MINUTES = 300

    # Minutes from trading start
    df = df.with_columns(
        ((pl.col("time_start_us") - TRADING_START_US) / 60_000_000).alias("_minutes_from_open")
    )

    # Cyclical encoding
    df = df.with_columns(
        [
            (2 * np.pi * pl.col("_minutes_from_open") / TRADING_MINUTES)
            .sin()
            .alias("time_sin"),
            (2 * np.pi * pl.col("_minutes_from_open") / TRADING_MINUTES)
            .cos()
            .alias("time_cos"),
        ]
    )

    # Session flags
    # Opening: first 5 minutes (9:00-9:05)
    df = df.with_columns(
        (pl.col("_minutes_from_open") < 5).cast(pl.Int32).alias("is_opening")
    )

    # Closing: last 10 minutes (14:50-15:00, roughly 290-300 minutes)
    df = df.with_columns(
        (pl.col("_minutes_from_open") > 290).cast(pl.Int32).alias("is_closing")
    )

    # Session indicator (1 = AM, 2 = PM)
    if "session" in df.columns:
        df = df.with_columns(
            pl.col("session").alias("session_indicator")
        )

    df = df.drop("_minutes_from_open")

    return df


def compute_all_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all technical indicator features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all technical features added
    """
    # RSI
    df = add_rsi(df, periods=[14, 20])

    # Bollinger Bands
    df = add_bollinger_bands(df, periods=[20], num_std=2.0)

    # MACD
    df = add_macd(df, fast_period=12, slow_period=26, signal_period=9)

    # Time features
    df = add_time_features(df)

    return df


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, "..")
    from data.bar_aggregator import load_bars_parquet

    bars = load_bars_parquet(
        "/Users/asefujiko/tools/tick_data/data/processed/toyota_7203_100tick_bars.parquet"
    )

    print("Computing technical features...")
    bars_with_features = compute_all_technical_features(bars)

    print(f"Columns: {bars_with_features.columns}")
    print(bars_with_features.head(5))
