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
    """Load stock prices from Bear Lake."""
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
    """Load stock returns from Bear Lake."""
    query = (
        bl.table("stock_returns")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    df = client.query(query)
    if tickers:
        df = df.filter(pl.col("ticker").is_in(tickers))
    return df.select(["ticker", "date", "return"]).sort("date", "ticker")


def load_universe(client: bl.Database, date: str) -> list[str]:
    """Load universe tickers for a specific date."""
    df = client.query(
        bl.table("universe").filter(pl.col("date") == pl.lit(date).str.to_date())
    )
    return df["ticker"].to_list()


def load_signals(
    client: bl.Database,
    start_date: str,
    end_date: str,
    signal_name: str = "reversal",
) -> pl.DataFrame:
    """Load signals from Bear Lake."""
    df = client.query(
        bl.table("signals")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
        .filter(pl.col("signal") == signal_name)
    )
    return df.select(["ticker", "date", "value"]).sort("date", "ticker")


def load_factor_loadings(
    client: bl.Database,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Load factor loadings for date range."""
    df = client.query(
        bl.table("factor_loadings")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df.select(["ticker", "date", "factor", "loading"]).sort("date", "ticker", "factor")


def load_factor_covariances(
    client: bl.Database,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Load factor covariance matrices for date range."""
    df = client.query(
        bl.table("factor_covariances")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df.select(["date", "factor_1", "factor_2", "covariance"]).sort("date")


def load_idio_vol(
    client: bl.Database,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Load idiosyncratic volatility for date range."""
    df = client.query(
        bl.table("idio_vol")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df.select(["ticker", "date", "idio_vol"]).sort("date", "ticker")


def load_benchmark_weights(
    client: bl.Database,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Load benchmark weights for date range."""
    df = client.query(
        bl.table("benchmark_weights")
        .filter(pl.col("date") >= pl.lit(start_date).str.to_date())
        .filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    )
    return df.select(["ticker", "date", "weight"]).sort("date", "ticker")
