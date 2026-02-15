from abc import ABC, abstractmethod
from malatium.data import FactorLoadingsProvider, FactorCovariancesProvider, IdioVolProvider
import datetime as dt
import numpy as np
import polars as pl


class RiskModel(ABC):
    """Base class for covariance matrix estimation."""

    @abstractmethod
    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> np.ndarray:
        """Build and return the asset covariance matrix for the given date and tickers."""
        pass


class FactorRiskModel(RiskModel):
    """Factor-based risk model: Sigma = X F X' + D^2.

    Computes the covariance matrix from factor loadings (X), a factor
    covariance matrix (F), and a diagonal matrix of idiosyncratic
    volatilities (D).
    """

    def __init__(
        self,
        factor_loadings: FactorLoadingsProvider,
        factor_covariances: FactorCovariancesProvider,
        idio_vol: IdioVolProvider,
    ):
        self.factor_loadings = factor_loadings
        self.factor_covariances = factor_covariances
        self.idio_vol = idio_vol

    def _build_factor_loadings_matrix(self, factor_loadings: pl.DataFrame, tickers: list[str]) -> np.ndarray:
        """Pivot factor loadings into an (n_assets x n_factors) numpy matrix."""
        return (
            factor_loadings.filter(pl.col("ticker").is_in(tickers))
            .sort("ticker", "factor")
            .pivot(index="ticker", on="factor", values="loading")
            .drop("ticker")
            .to_numpy()
        )

    def _build_factor_covariance_matrix(self, factor_covariances: pl.DataFrame) -> np.ndarray:
        """Pivot factor covariances into a symmetric (n_factors x n_factors) numpy matrix."""
        return (
            factor_covariances.sort("factor_1", "factor_2")
            .pivot(index="factor_1", on="factor_2", values="covariance")
            .drop("factor_1")
            .to_numpy()
        )

    def _build_idio_vol_matrix(self, idio_vol: pl.DataFrame, tickers: list[str]) -> np.ndarray:
        """Build a diagonal matrix of idiosyncratic volatilities."""
        return np.diag(
            idio_vol.filter(pl.col("ticker").is_in(tickers))
            .sort("ticker")["idio_vol"]
            .to_numpy()
        )

    def build_covariance_matrix(self, date_: dt.date, tickers: list[str]) -> np.ndarray:
        """Compute the full covariance matrix as X @ F @ X.T + D^2."""
        factor_loadings = self.factor_loadings.get(date_)
        factor_covariances = self.factor_covariances.get(date_)
        idio_vol = self.idio_vol.get(date_)

        X = self._build_factor_loadings_matrix(factor_loadings, tickers)
        F = self._build_factor_covariance_matrix(factor_covariances)
        D = self._build_idio_vol_matrix(idio_vol, tickers)

        return X @ F @ X.T + D ** 2
