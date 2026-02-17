"""Data providers for connecting to Bear Lake and loading market data."""
import os
import datetime as dt
import bear_lake as bl
import polars as pl
from atium.models import Alphas, BenchmarkWeights, Returns, FactorLoadings, FactorCovariances, IdioVol


def get_bear_lake_client():
    access_key_id = os.getenv("ACCESS_KEY_ID")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    region = os.getenv("REGION")
    endpoint = os.getenv("ENDPOINT")
    bucket = os.getenv("BUCKET")

    storage_options = {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
        "region": region,
        "endpoint_url": endpoint,
    }

    url = f"s3://{bucket}"

    return bl.connect_s3(path=url, storage_options=storage_options)


class MyCalendarProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('calendar')
            .filter(pl.col('date').is_between(start, end))
            .sort('date')
        )

    def get(self, start: dt.date, end: dt.date) -> list[dt.date]:
        return self.data.filter(pl.col('date').is_between(start, end))['date'].unique().sort().to_list()


class MyReturnsProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('stock_returns')
            .sort('ticker', 'date')
            .with_columns(pl.col('return').shift(-1).over('ticker'))
            .filter(pl.col('date').is_between(start, end))
            .drop('year')
            .sort('date', 'ticker')
        )

    def get(self, date_: dt.date) -> Returns:
        return Returns.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker'))


class MyAlphaProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('alphas')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('alpha').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def get(self, date_: dt.date) -> Alphas:
        return Alphas.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker'))


class MyFactorLoadingsProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('factor_loadings')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('loading').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker', 'factor')
        )

    def get(self, date_: dt.date) -> FactorLoadings:
        return FactorLoadings.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker', 'factor'))


class MyFactorCovariancesProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('factor_covariances')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('covariance').is_not_null()
            )
            .drop('year')
            .sort('date', 'factor_1', 'factor_2')
        )

    def get(self, date_: dt.date) -> FactorCovariances:
        return FactorCovariances.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'factor_1', 'factor_2'))


class MyIdioVolProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('idio_vol')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('idio_vol').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def get(self, date_: dt.date) -> IdioVol:
        return IdioVol.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker'))


class MyBenchmarkWeightsProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> BenchmarkWeights:
        self.data = db.query(
            bl.table('benchmark_weights')
            .filter(pl.col('date').is_between(start, end))
            .drop('year')
            .sort('date', 'ticker')
        )

    def get(self, date_: dt.date) -> BenchmarkWeights:
        return BenchmarkWeights.validate(self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker'))
