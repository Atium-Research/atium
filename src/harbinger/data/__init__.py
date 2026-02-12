"""Data adapters and schemas."""

import os
from pathlib import Path

import bear_lake as bl
import polars as pl
from dotenv import load_dotenv


def get_bear_lake_client(env_path: str | Path | None = None) -> bl.Database:
    """Create Bear Lake client from environment variables."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

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
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """
    Load stock prices from Bear Lake.
    
    Returns DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    query = (
        bl.table("stock_prices")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    
    df = client.query(query)
    
    if tickers:
        df = df.filter(pl.col("ticker").is_in(tickers))
    
    return df.select(["ticker", "date", "open", "high", "low", "close", "volume"]).sort("date", "ticker")


def load_returns(
    client: bl.Database,
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """
    Load stock returns from Bear Lake.
    
    Returns DataFrame with columns: ticker, date, return
    """
    query = (
        bl.table("stock_returns")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    
    df = client.query(query)
    
    if tickers:
        df = df.filter(pl.col("ticker").is_in(tickers))
    
    return df.select(["ticker", "date", "return"]).sort("date", "ticker")


def load_universe(
    client: bl.Database,
    date: str,
) -> list[str]:
    """Load universe tickers for a specific date."""
    df = client.query(
        bl.table("universe").filter(pl.col("date") == pl.lit(date).str.to_date())
    )
    return df["ticker"].to_list()


def returns_to_matrix(returns: pl.DataFrame) -> tuple[pl.DataFrame, list[str], list]:
    """
    Convert long-format returns to wide matrix.
    
    Returns:
        - matrix: DataFrame with dates as rows, tickers as columns
        - tickers: list of ticker names (column order)
        - dates: list of dates (row order)
    """
    wide = returns.pivot(on="ticker", index="date", values="return").sort("date")
    tickers = [c for c in wide.columns if c != "date"]
    dates = wide["date"].to_list()
    return wide, tickers, dates
