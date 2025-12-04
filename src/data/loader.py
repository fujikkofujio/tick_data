"""
Tick Data Loader Module

Handles loading and parsing of TSE tick data CSV files.
Optimized for large files (8GB+) using Polars for efficient memory usage.
"""

import polars as pl
from pathlib import Path
from typing import Optional


# Column names matching the CSV structure
TICK_COLUMNS = [
    "date",
    "issue_code",
    "isin_code",
    "exchange_code",
    "issue_classification",
    "industry_code",
    "supervision_flag",
    "time",
    "session",
    "price",
    "volume",
    "transaction_id",
]

# Data types for efficient parsing (using original CSV column names with spaces)
TICK_DTYPES = {
    "date": pl.Utf8,
    "issue code": pl.Utf8,
    "isin code": pl.Utf8,
    "exchange code": pl.Utf8,
    "issue classification": pl.Utf8,
    "industry code": pl.Utf8,
    "securities under supervision and to be delisted flag": pl.Utf8,
    "time": pl.Utf8,
    "session distinction": pl.Utf8,
    "price": pl.Float64,
    "trading volume": pl.Int64,
    "transaction id": pl.Utf8,
}


def load_tick_data(
    csv_path: str | Path,
    stock_code: Optional[str] = None,
    n_rows: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load tick data from CSV file.

    Args:
        csv_path: Path to the CSV file
        stock_code: Optional stock code to filter (e.g., "72030" for Toyota)
        n_rows: Optional limit on number of rows to read

    Returns:
        Polars DataFrame with tick data
    """
    csv_path = Path(csv_path)

    # Use lazy evaluation for efficient filtering
    lf = pl.scan_csv(
        csv_path,
        has_header=True,
        dtypes=TICK_DTYPES,
        n_rows=n_rows,
    )

    # Rename columns to standardized names
    lf = lf.rename(
        {
            "date": "date",
            "issue code": "issue_code",
            "isin code": "isin_code",
            "exchange code": "exchange_code",
            "issue classification": "issue_classification",
            "industry code": "industry_code",
            "securities under supervision and to be delisted flag": "supervision_flag",
            "time": "time",
            "session distinction": "session",
            "price": "price",
            "trading volume": "volume",
            "transaction id": "transaction_id",
        }
    )

    # Filter by stock code if specified
    if stock_code:
        # Handle both trimmed and padded stock codes
        lf = lf.filter(pl.col("issue_code").str.strip_chars().eq(stock_code))

    # Collect and return
    df = lf.collect()

    return df


def preprocess_tick_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess tick data for analysis.

    Adds computed columns:
    - datetime: Parsed timestamp
    - stock_code: Trimmed issue code
    - time_microseconds: Time as microseconds from midnight

    Args:
        df: Raw tick DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df = df.with_columns(
        [
            # Trim stock code
            pl.col("issue_code").str.strip_chars().alias("stock_code"),
            # Parse date
            pl.col("date")
            .str.strptime(pl.Date, format="%Y%m%d")
            .alias("trade_date"),
            # Parse time components from hhmmsstttttt format
            pl.col("time")
            .str.slice(0, 2)
            .cast(pl.Int32)
            .alias("hour"),
            pl.col("time")
            .str.slice(2, 2)
            .cast(pl.Int32)
            .alias("minute"),
            pl.col("time")
            .str.slice(4, 2)
            .cast(pl.Int32)
            .alias("second"),
            pl.col("time")
            .str.slice(6, 6)
            .cast(pl.Int64)
            .alias("microsecond"),
            # Session as integer
            pl.col("session").cast(pl.Int32).alias("session_int"),
        ]
    )

    # Calculate total microseconds from midnight
    df = df.with_columns(
        (
            pl.col("hour") * 3600_000_000
            + pl.col("minute") * 60_000_000
            + pl.col("second") * 1_000_000
            + pl.col("microsecond")
        ).alias("time_us")
    )

    return df


def load_and_preprocess(
    csv_path: str | Path,
    stock_code: str,
    n_rows: Optional[int] = None,
) -> pl.DataFrame:
    """
    Convenience function to load and preprocess tick data in one step.

    Args:
        csv_path: Path to the CSV file
        stock_code: Stock code to filter (e.g., "72030" for Toyota)
        n_rows: Optional limit on number of rows

    Returns:
        Preprocessed DataFrame
    """
    df = load_tick_data(csv_path, stock_code=stock_code, n_rows=n_rows)
    df = preprocess_tick_data(df)
    return df


if __name__ == "__main__":
    # Test loading
    import sys

    csv_path = Path("/Users/asefujiko/tools/tick_data/stock_tick_data/stock_tick_202510.csv")

    print("Loading Toyota (7203) tick data...")
    df = load_and_preprocess(csv_path, stock_code="72030", n_rows=1000)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns}")
    print(df.head(5))
