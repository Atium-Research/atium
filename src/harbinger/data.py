from abc import ABC, abstractmethod
import datetime as dt
import bear_lake as bl
import os
import polars as pl

class DataAdapter(ABC):

    @abstractmethod
    def get_calendar(self, start: dt.date, end: dt.date) -> list[dt.date]:
        pass

    @abstractmethod
    def get_universe(self, date_: dt.date) -> list[str]:
        pass

    @abstractmethod
    def get_prices(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_returns(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_benchmark_weights(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_signals(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_scores(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_alphas(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_betas(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_factor_loadings(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_factor_covariances(self, date_: dt.date) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_idio_vol(self, date_: dt.date) -> pl.DataFrame:
        pass
