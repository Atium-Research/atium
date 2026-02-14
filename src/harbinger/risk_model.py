from abc import ABC, abstractmethod
import polars as pl
from harbinger.data import DataAdapter
import datetime as dt
import numpy as np

class RiskModel(ABC):
    @abstractmethod
    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> pl.DataFrame:
        pass

class FactorRiskModel(RiskModel):
    def __init__(self, data: DataAdapter):
        self.data = data

    def _build_factor_loadings_matrix(self, factor_loadings: pl.DataFrame, tickers: list[str]) -> np.ndarray:
        return (
            factor_loadings.filter(pl.col("ticker").is_in(tickers))
            .sort("ticker", "factor")
            .pivot(index="ticker", on="factor", values="loading")
            .drop("ticker")
            .to_numpy()
        )
    
    def _build_factor_covariance_matrix(self, factor_covariances: pl.DataFrame) -> np.ndarray:
        return (
            factor_covariances.sort("factor_1", "factor_2")
            .pivot(index="factor_1", on="factor_2", values="covariance")
            .drop("factor_1")
            .to_numpy()
        )


    def _build_idio_vol_matrix(self, idio_vol: pl.DataFrame, tickers: list[str]) -> np.ndarray:
        return np.diag(
            idio_vol.filter(pl.col("ticker").is_in(tickers))
            .sort("ticker")["idio_vol"]
            .to_numpy()
        )


    def _construct_covariance_matrix(
        self,
        factor_loadings_matrix: np.ndarray,
        factor_covariance_matrix: np.ndarray,
        idio_vol_matrix: np.ndarray,
        tickers: list[str],
    ) -> pl.DataFrame:
        covariance_matrix_np = (
            factor_loadings_matrix @ factor_covariance_matrix @ factor_loadings_matrix.T
            + idio_vol_matrix**2
        )

        covariance_matrix = pl.from_numpy(covariance_matrix_np)
        covariance_matrix.columns = tickers
        covariance_matrix = covariance_matrix.select(
            pl.Series(tickers).alias("ticker"), *tickers
        )

        return covariance_matrix

    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> pl.DataFrame:
        factor_loadings = self.data.get_factor_loadings(date_)
        factor_covariances = self.data.get_factor_covariances(date_)
        idio_vol = self.data.get_idio_vol(date_)

        factor_loadings_matrix = self._build_factor_loadings_matrix(factor_loadings, tickers)
        factor_covariance_matrix = self._build_factor_covariance_matrix(factor_covariances)
        idio_vol_matrix = self._build_idio_vol_matrix(idio_vol, tickers)

        covariance_matrix = self._construct_covariance_matrix(
            factor_loadings_matrix=factor_loadings_matrix,
            factor_covariance_matrix=factor_covariance_matrix,
            idio_vol_matrix=idio_vol_matrix,
            tickers=tickers
        )

        return covariance_matrix

