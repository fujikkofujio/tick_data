"""
Preprocessing Script

Prepare tick data for RL training:
1. Load raw CSV
2. Extract single stock
3. Aggregate to 100-tick bars
4. Compute features
5. Save to Parquet and NumPy
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from datetime import datetime

from data.loader import load_and_preprocess
from data.bar_aggregator import aggregate_tick_bars, save_bars_parquet
from data.preprocessor import prepare_training_data


def main(
    csv_path: str,
    stock_code: str,
    output_dir: str = None,
    bar_size: int = 100,
    warmup_bars: int = 100,
):
    """
    Full preprocessing pipeline.

    Args:
        csv_path: Path to raw tick CSV
        stock_code: Stock code to extract (e.g., "72030" for Toyota)
        output_dir: Output directory
        bar_size: Ticks per bar
        warmup_bars: Number of bars to skip for feature warmup
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tick Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"CSV Path: {csv_path}")
    print(f"Stock Code: {stock_code}")
    print(f"Bar Size: {bar_size}")
    print(f"Output Dir: {output_dir}")
    print()

    # Step 1: Load raw tick data
    print("Step 1: Loading tick data...")
    df = load_and_preprocess(csv_path, stock_code=stock_code)
    print(f"  Loaded {len(df):,} ticks")

    # Step 2: Aggregate to bars
    print(f"\nStep 2: Aggregating to {bar_size}-tick bars...")
    bars = aggregate_tick_bars(df, bar_size=bar_size)
    print(f"  Created {len(bars):,} bars")

    # Step 3: Save bars
    bar_path = output_dir / f"stock_{stock_code}_{bar_size}tick_bars.parquet"
    save_bars_parquet(bars, str(bar_path))
    print(f"  Saved bars to: {bar_path}")

    # Step 4: Compute features and extract state vector
    print(f"\nStep 3: Computing features (warmup={warmup_bars})...")
    processed_df, state_array = prepare_training_data(
        bar_path=str(bar_path),
        output_dir=str(output_dir),
        warmup_bars=warmup_bars,
    )

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Raw ticks: {len(df):,}")
    print(f"  {bar_size}-tick bars: {len(bars):,}")
    print(f"  Training samples: {len(state_array):,}")
    print(f"  Feature dimensions: {state_array.shape[1]}")
    print(f"\nOutput files:")
    print(f"  - {bar_path}")
    print(f"  - {output_dir / 'processed_features.parquet'}")
    print(f"  - {output_dir / 'state_array.npy'}")

    return state_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tick data for RL")
    parser.add_argument(
        "--csv",
        type=str,
        default="/Users/asefujiko/tools/tick_data/stock_tick_data/stock_tick_202510.csv",
        help="Path to raw tick CSV",
    )
    parser.add_argument(
        "--stock",
        type=str,
        default="72030",  # Toyota
        help="Stock code to extract",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--bar-size",
        type=int,
        default=100,
        help="Ticks per bar",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup bars to skip",
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        stock_code=args.stock,
        output_dir=args.output,
        bar_size=args.bar_size,
        warmup_bars=args.warmup,
    )
