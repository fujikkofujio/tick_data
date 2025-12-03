"""
Data Preprocessor Module

Combines all feature computation and normalization for RL training.
Produces the final state vector ready for the trading environment.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.price_features import compute_all_price_features
from features.volume_features import compute_all_volume_features
from features.microstructure import compute_all_microstructure_features
from features.technical import compute_all_technical_features


# Default state vector columns for RL
DEFAULT_STATE_COLUMNS = [
    # Price dynamics
    "return_1",
    "return_5",
    "return_20",
    "return_100",
    "price_position_20",
    "price_position_100",
    # Volatility
    "realized_vol_20",
    "realized_vol_100",
    "vol_ratio",
    # Volume
    "volume_ratio_20",
    "volume_ratio_100",
    "volume_zscore",
    "vwap_deviation_20",
    "vwap_deviation_100",
    "dollar_volume_normalized",
    # Order flow
    "ofi_normalized_20",
    "ofi_normalized_100",
    "cum_sign_normalized_20",
    "trade_sign",
    # Technical
    "rsi_20_normalized",
    "bb_position_20",
    "bb_width_20",
    "macd_histogram_normalized",
    # Time
    "time_sin",
    "time_cos",
    "is_opening",
    "is_closing",
]


def compute_all_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all features from bar data.

    Args:
        df: DataFrame with OHLCV bar data

    Returns:
        DataFrame with all features added
    """
    print("Computing price features...")
    df = compute_all_price_features(df)

    print("Computing volume features...")
    df = compute_all_volume_features(df)

    print("Computing microstructure features...")
    df = compute_all_microstructure_features(df)

    print("Computing technical features...")
    df = compute_all_technical_features(df)

    return df


def normalize_features(
    df: pl.DataFrame,
    columns: List[str],
    method: str = "zscore",
    clip_range: Tuple[float, float] = (-5.0, 5.0),
) -> Tuple[pl.DataFrame, Dict]:
    """
    Normalize feature columns.

    Args:
        df: DataFrame with features
        columns: Columns to normalize
        method: Normalization method ('zscore', 'minmax', 'rank')
        clip_range: Value range to clip after normalization

    Returns:
        Tuple of (normalized DataFrame, normalization parameters)
    """
    norm_params = {}

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue

        if method == "zscore":
            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val is None or std_val == 0:
                std_val = 1.0

            norm_params[col] = {"method": "zscore", "mean": mean_val, "std": std_val}

            df = df.with_columns(
                ((pl.col(col) - mean_val) / std_val)
                .clip(clip_range[0], clip_range[1])
                .alias(f"{col}_norm")
            )

        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()

            if max_val == min_val:
                max_val = min_val + 1.0

            norm_params[col] = {"method": "minmax", "min": min_val, "max": max_val}

            df = df.with_columns(
                ((pl.col(col) - min_val) / (max_val - min_val))
                .clip(0.0, 1.0)
                .alias(f"{col}_norm")
            )

    return df, norm_params


def extract_state_vector(
    df: pl.DataFrame,
    columns: List[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract state vector as numpy array for RL environment.

    Args:
        df: DataFrame with computed features
        columns: Columns to include in state vector (default: DEFAULT_STATE_COLUMNS)
        normalize: Whether to apply z-score normalization

    Returns:
        numpy array of shape (n_bars, n_features)
    """
    if columns is None:
        columns = DEFAULT_STATE_COLUMNS

    # Filter to available columns
    available_cols = [c for c in columns if c in df.columns]
    missing_cols = [c for c in columns if c not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    # Select and convert
    state_df = df.select(available_cols)

    if normalize:
        state_df, _ = normalize_features(state_df, available_cols)
        # Use normalized columns
        norm_cols = [f"{c}_norm" for c in available_cols if f"{c}_norm" in state_df.columns]
        if norm_cols:
            state_df = state_df.select(norm_cols)

    # Convert to numpy, filling NaN with 0
    state_array = state_df.to_numpy()
    state_array = np.nan_to_num(state_array, nan=0.0, posinf=5.0, neginf=-5.0)

    return state_array


def prepare_training_data(
    bar_path: str,
    output_dir: str = None,
    warmup_bars: int = 100,
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Full pipeline to prepare data for RL training.

    Args:
        bar_path: Path to bar data (parquet)
        output_dir: Optional output directory to save processed data
        warmup_bars: Number of initial bars to skip (for warmup of rolling features)

    Returns:
        Tuple of (processed DataFrame, state array)
    """
    print(f"Loading bars from {bar_path}...")
    df = pl.read_parquet(bar_path)
    print(f"Loaded {len(df)} bars")

    # Compute features
    df = compute_all_features(df)
    print(f"Computed {len(df.columns)} columns")

    # Skip warmup period
    df = df.slice(warmup_bars, len(df) - warmup_bars)
    print(f"After warmup skip: {len(df)} bars")

    # Extract state vector
    state_array = extract_state_vector(df)
    print(f"State array shape: {state_array.shape}")

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed DataFrame
        df.write_parquet(output_dir / "processed_features.parquet")
        print(f"Saved features to {output_dir / 'processed_features.parquet'}")

        # Save state array
        np.save(output_dir / "state_array.npy", state_array)
        print(f"Saved state array to {output_dir / 'state_array.npy'}")

    return df, state_array


def get_feature_names() -> List[str]:
    """Return the default state vector feature names."""
    return DEFAULT_STATE_COLUMNS.copy()


if __name__ == "__main__":
    # Test full pipeline
    bar_path = "/Users/asefujiko/tools/tick_data/data/processed/toyota_7203_100tick_bars.parquet"
    output_dir = "/Users/asefujiko/tools/tick_data/data/processed"

    df, state_array = prepare_training_data(bar_path, output_dir)

    print("\nFinal state vector info:")
    print(f"  Shape: {state_array.shape}")
    print(f"  Features: {len(get_feature_names())}")
    print(f"  Min: {state_array.min():.4f}")
    print(f"  Max: {state_array.max():.4f}")
    print(f"  Mean: {state_array.mean():.4f}")
