"""
Bar Aggregator Module

Converts raw tick data into OHLCV bars (tick bars, time bars, volume bars).
Primary focus: 100-tick bars for RL training.
"""

import polars as pl
import numpy as np
from typing import Literal


def aggregate_tick_bars(
    df: pl.DataFrame,
    bar_size: int = 100,
    include_metadata: bool = True,
) -> pl.DataFrame:
    """
    Aggregate tick data into N-tick bars.

    Each bar represents N consecutive ticks and contains:
    - OHLCV (Open, High, Low, Close, Volume)
    - Tick count
    - Time span
    - Optional metadata (start/end times, etc.)

    Args:
        df: Preprocessed tick DataFrame (must have price, volume, time_us columns)
        bar_size: Number of ticks per bar (default: 100)
        include_metadata: Whether to include additional metadata columns

    Returns:
        DataFrame with aggregated bars
    """
    # Ensure data is sorted by time
    df = df.sort(["trade_date", "time_us"])

    # Add bar index
    df = df.with_columns(
        (pl.arange(0, pl.len()) // bar_size).alias("bar_idx")
    )

    # Aggregate by bar
    bars = df.group_by("bar_idx", maintain_order=True).agg(
        [
            # OHLCV
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            # Tick count in this bar
            pl.len().alias("tick_count"),
            # Time info
            pl.col("trade_date").first().alias("date"),
            pl.col("time_us").first().alias("time_start_us"),
            pl.col("time_us").last().alias("time_end_us"),
            pl.col("session_int").first().alias("session"),
            # Price dynamics within bar
            pl.col("price").mean().alias("vwap_simple"),  # Simple mean, not true VWAP
            pl.col("price").std().alias("price_std"),
            # Volume dynamics
            pl.col("volume").mean().alias("volume_mean"),
            pl.col("volume").max().alias("volume_max"),
        ]
    )

    # Calculate true VWAP (volume-weighted average price)
    # Need to recalculate from original data
    df_with_dollar = df.with_columns(
        (pl.col("price") * pl.col("volume")).alias("dollar_volume")
    )

    vwap_df = df_with_dollar.group_by("bar_idx", maintain_order=True).agg(
        [
            pl.col("dollar_volume").sum().alias("total_dollar_volume"),
            pl.col("volume").sum().alias("total_volume"),
        ]
    )

    vwap_df = vwap_df.with_columns(
        (pl.col("total_dollar_volume") / pl.col("total_volume")).alias("vwap")
    ).select(["bar_idx", "vwap"])

    bars = bars.join(vwap_df, on="bar_idx", how="left")

    # Add derived features
    bars = bars.with_columns(
        [
            # Bar range
            (pl.col("high") - pl.col("low")).alias("range"),
            # Bar return
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("bar_return"),
            # Time duration (microseconds)
            (pl.col("time_end_us") - pl.col("time_start_us")).alias("duration_us"),
            # Dollar volume
            (pl.col("vwap") * pl.col("volume")).alias("dollar_volume"),
        ]
    )

    if not include_metadata:
        bars = bars.select(
            ["bar_idx", "date", "open", "high", "low", "close", "volume", "bar_return"]
        )

    return bars


def aggregate_time_bars(
    df: pl.DataFrame,
    interval_seconds: int = 60,
) -> pl.DataFrame:
    """
    Aggregate tick data into time-based bars.

    Args:
        df: Preprocessed tick DataFrame
        interval_seconds: Bar interval in seconds (default: 60 = 1 minute)

    Returns:
        DataFrame with time bars
    """
    interval_us = interval_seconds * 1_000_000

    # Calculate time bar index
    df = df.with_columns(
        (pl.col("time_us") // interval_us).alias("time_bar_idx")
    )

    # Aggregate by time bar
    bars = df.group_by(["trade_date", "time_bar_idx"], maintain_order=True).agg(
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("tick_count"),
            pl.col("time_us").first().alias("time_start_us"),
            pl.col("session_int").first().alias("session"),
        ]
    )

    bars = bars.with_columns(
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("bar_return")
    )

    return bars


def aggregate_volume_bars(
    df: pl.DataFrame,
    volume_threshold: int = 10000,
) -> pl.DataFrame:
    """
    Aggregate tick data into volume-based bars.

    A new bar starts when cumulative volume exceeds the threshold.

    Args:
        df: Preprocessed tick DataFrame
        volume_threshold: Volume threshold per bar

    Returns:
        DataFrame with volume bars
    """
    # Sort by time
    df = df.sort(["trade_date", "time_us"])

    # Calculate cumulative volume and bar boundaries
    # This requires iterating, so we convert to numpy for efficiency
    volumes = df["volume"].to_numpy()

    bar_indices = []
    current_bar = 0
    cumulative_vol = 0

    for vol in volumes:
        cumulative_vol += vol
        if cumulative_vol >= volume_threshold:
            current_bar += 1
            cumulative_vol = 0
        bar_indices.append(current_bar)

    df = df.with_columns(pl.Series("volume_bar_idx", bar_indices))

    # Aggregate
    bars = df.group_by("volume_bar_idx", maintain_order=True).agg(
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("tick_count"),
            pl.col("trade_date").first().alias("date"),
            pl.col("time_us").first().alias("time_start_us"),
            pl.col("time_us").last().alias("time_end_us"),
            pl.col("session_int").first().alias("session"),
        ]
    )

    bars = bars.with_columns(
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("bar_return")
    )

    return bars


def save_bars_parquet(
    bars: pl.DataFrame,
    output_path: str,
) -> None:
    """Save bar data to Parquet format for efficient loading."""
    bars.write_parquet(output_path)
    print(f"Saved {len(bars)} bars to {output_path}")


def load_bars_parquet(path: str) -> pl.DataFrame:
    """Load bar data from Parquet file."""
    return pl.read_parquet(path)


if __name__ == "__main__":
    from loader import load_and_preprocess
    from pathlib import Path

    # Test aggregation
    csv_path = Path("/Users/asefujiko/tools/tick_data/stock_tick_data/stock_tick_202510.csv")
    output_dir = Path("/Users/asefujiko/tools/tick_data/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Toyota (7203) tick data...")
    df = load_and_preprocess(csv_path, stock_code="72030")
    print(f"Loaded {len(df)} ticks")

    print("\nAggregating to 100-tick bars...")
    bars_100 = aggregate_tick_bars(df, bar_size=100)
    print(f"Created {len(bars_100)} bars")
    print(bars_100.head(5))

    # Save to parquet
    output_path = output_dir / "toyota_7203_100tick_bars.parquet"
    save_bars_parquet(bars_100, str(output_path))
