import datetime as dt
import bear_lake as bl
import os
import polars as pl
from harbinger.data import DataAdapter


class MyDataAdapter(DataAdapter):
    def __init__(self, start: dt.date, end: dt.date) -> None:

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

        self.db = bl.connect_s3(path=url, storage_options=storage_options)

        print("LOADING DATA...")
        self.calendar = self._load_calendar(start, end)
        self.universe = self._load_universe(start, end)
        self.prices = self._load_prices(start, end)
        self.forward_returns = self._load_forward_returns(start, end)
        self.benchmark_weights = self._load_benchmark_weights(start, end)
        self.signals = self._load_signals(start, end)
        self.scores = self._load_scores(start, end)
        self.alphas = self._load_alphas(start, end)
        self.betas = self._load_betas(start, end)
        self.factor_loadings = self._load_factor_loadings(start, end)
        self.factor_covariances = self._load_factor_covariances(start, end)
        self.idio_vol = self._load_idio_vol(start, end)

    def _load_calendar(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('calendar')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .sort('date')
        )

    def _load_universe(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('universe')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_prices(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('stock_prices')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_forward_returns(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('stock_returns')
            .sort('ticker', 'date')
            .with_columns(
                pl.col('return').shift(-1).over('ticker')
            )
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_benchmark_weights(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('benchmark_weights')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_betas(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('betas')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_signals(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('signals')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_scores(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('scores')
            .filter(
                pl.col('date').is_between(start, end)
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_alphas(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('alphas')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('alpha').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker')
        )

    def _load_factor_loadings(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('factor_loadings')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('loading').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker', 'factor')
        )

    def _load_factor_covariances(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('factor_covariances')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('covariance').is_not_null()
            )
            .drop('year')
            .sort('date', 'factor_1', 'factor_2')
        )

    def _load_idio_vol(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        return self.db.query(
            bl.table('idio_vol')
            .filter(
                pl.col('date').is_between(start, end),
                pl.col('idio_vol').is_not_null()
            )
            .drop('year')
            .sort('date', 'ticker')
        )
    
    def get_calendar(self, start: dt.date, end: dt.date) -> list[dt.date]:
        return self.calendar.filter(pl.col('date').is_between(start, end))['date'].unique().sort().to_list()

    def get_universe(self, date_: dt.date) -> list[str]:
        return self.universe.filter(pl.col('date').eq(date_))['ticker'].unique().sort().to_list()

    def get_prices(self, date_: dt.date) -> pl.DataFrame:
        return self.prices.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_forward_returns(self, date_: dt.date) -> pl.DataFrame:
        return self.forward_returns.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_benchmark_weights(self, date_: dt.date) -> pl.DataFrame:
        return self.benchmark_weights.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_signals(self, date_: dt.date) -> pl.DataFrame:
        return self.signals.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_scores(self, date_: dt.date) -> pl.DataFrame:
        return self.scores.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_alphas(self, date_: dt.date) -> pl.DataFrame:
        return self.alphas.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_betas(self, date_: dt.date) -> pl.DataFrame:
        return self.betas.filter(pl.col('date').eq(date_)).sort('date', 'ticker')

    def get_factor_loadings(self, date_: dt.date) -> pl.DataFrame:
        return self.factor_loadings.filter(pl.col('date').eq(date_)).sort('date', 'ticker', 'factor')

    def get_factor_covariances(self, date_: dt.date) -> pl.DataFrame:
        return self.factor_covariances.filter(pl.col('date').eq(date_)).sort('date', 'factor_1', 'factor_2')

    def get_idio_vol(self, date_: dt.date) -> pl.DataFrame:
        return self.idio_vol.filter(pl.col('date').eq(date_)).sort('date', 'ticker')
