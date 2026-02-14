from abc import ABC, abstractmethod
from harbinger.data import RiskDataProvider
import datetime as dt
import numpy as np
import polars as pl


class RiskModel(ABC):
    @abstractmethod
    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> np.ndarray:
        pass


class FactorRiskModel(RiskModel):
    def __init__(self, data: RiskDataProvider):
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

    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> np.ndarray:
        factor_loadings = self.data.get_factor_loadings(date_)
        factor_covariances = self.data.get_factor_covariances(date_)
        idio_vol = self.data.get_idio_vol(date_)

        X = self._build_factor_loadings_matrix(factor_loadings, tickers)
        F = self._build_factor_covariance_matrix(factor_covariances)
        D = self._build_idio_vol_matrix(idio_vol, tickers)

        return X @ F @ X.T + D ** 2
