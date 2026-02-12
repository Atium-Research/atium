"""
Test harbinger functionality with real data from Bear Lake.

Usage:
    uv run pytest tests/test_with_data.py -v
    uv run python tests/test_with_data.py  # For interactive exploration
"""

import os
from pathlib import Path

import bear_lake as bl
import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


def get_bear_lake_client() -> bl.Database:
    """Create Bear Lake client from environment variables."""
    storage_options = {
        "aws_access_key_id": os.environ["ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["SECRET_ACCESS_KEY"],
        "region": os.environ["REGION"],
        "endpoint_url": os.environ["ENDPOINT"],
    }
    url = f"s3://{os.environ['BUCKET']}"
    return bl.connect_s3(path=url, storage_options=storage_options)


def load_prices(
    client: bl.Database,
    start_date: str = "2026-01-01",
    end_date: str = "2026-02-01",
) -> pl.DataFrame:
    """Load stock prices from Bear Lake."""
    df = client.query(
        bl.table("stock_prices")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df


def load_returns(
    client: bl.Database,
    start_date: str = "2026-01-01",
    end_date: str = "2026-02-01",
) -> pl.DataFrame:
    """Load stock returns from Bear Lake."""
    df = client.query(
        bl.table("stock_returns")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df


def load_universe(client: bl.Database, date: str = "2026-01-02") -> pl.DataFrame:
    """Load universe from Bear Lake."""
    df = client.query(
        bl.table("universe").filter(pl.col("date") == pl.lit(date).str.to_date())
    )
    return df


# ============================================================
# Tests
# ============================================================


def test_connection():
    """Test Bear Lake connection."""
    client = get_bear_lake_client()
    tables = client.list_tables()
    assert len(tables) > 0, "No tables found"
    print(f"✓ Connected to Bear Lake, found {len(tables)} tables")


def test_load_prices():
    """Test loading price data."""
    client = get_bear_lake_client()
    df = load_prices(client, "2026-01-01", "2026-01-31")

    assert not df.is_empty(), "No price data"
    assert "date" in df.columns
    assert "ticker" in df.columns
    assert "close" in df.columns

    print(f"✓ Loaded prices: {df.shape}")
    print(df.head())


def test_load_universe():
    """Test loading universe."""
    client = get_bear_lake_client()
    df = load_universe(client, "2026-01-02")

    assert not df.is_empty(), "No universe data"
    assert "ticker" in df.columns

    print(f"✓ Loaded universe: {len(df)} tickers")


# ============================================================
# Interactive mode
# ============================================================

if __name__ == "__main__":
    print("\n🔮 Harbinger — Data Exploration\n")

    client = get_bear_lake_client()

    print("=" * 60)
    print("Bear Lake Tables")
    print("=" * 60)

    tables = client.list_tables()
    for table in tables:
        print(f"  📊 {table}")

    print("\n" + "=" * 60)
    print("Sample Data")
    print("=" * 60)

    # Load prices
    print("\n📈 Stock Prices (Jan 2026):")
    prices = load_prices(client, "2026-01-01", "2026-01-31")
    print(f"   Shape: {prices.shape}")
    print(f"   Tickers: {prices['ticker'].n_unique()}")
    print(f"   Date range: {prices['date'].min()} to {prices['date'].max()}")
    print(prices.head(5))

    # Load returns
    print("\n📊 Stock Returns (Jan 2026):")
    returns = load_returns(client, "2026-01-01", "2026-01-31")
    print(f"   Shape: {returns.shape}")
    print(returns.head(5))

    # Load universe
    print("\n🌐 Universe (2026-01-02):")
    universe = load_universe(client, "2026-01-02")
    print(f"   Tickers: {len(universe)}")
    print(universe.head(5))

    print("\n✅ Data exploration complete!")
