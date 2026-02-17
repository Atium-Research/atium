import datetime as dt
import bear_lake as bl
import polars as pl


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

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker')


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

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker')


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

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker', 'factor')


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

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'factor_1', 'factor_2')


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

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker')


class MyBenchmarkWeightsProvider:
    def __init__(self, db: bl.Database, start: dt.date, end: dt.date) -> None:
        self.data = db.query(
            bl.table('benchmark_weights')
            .filter(pl.col('date').is_between(start, end))
            .drop('year')
            .sort('date', 'ticker')
        )

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self.data.filter(pl.col('date').eq(date_)).sort('date', 'ticker')
