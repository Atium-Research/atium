from abc import ABC, abstractmethod
from atium.data import FactorLoadingsProvider, FactorCovariancesProvider, IdioVolProvider
from atium.models import FactorLoadings, FactorCovariances, IdioVol
import datetime as dt
import numpy as np
import polars as pl

class RiskModel(ABC):
    """Base class for covariance matrix estimation."""

    @abstractmethod
    def build_covariance_matrix(self) -> np.ndarray:
        """Build and return the asset covariance matrix for the given date."""
        pass

    @property
    @abstractmethod
    def tickers(self) -> list[str]:
        pass

class RiskModelConstructor(ABC):
    """Base class for covariance matrix estimation."""

    @abstractmethod
    def get_risk_model(self, date_: dt.date) -> RiskModel:
        pass

class FactorRiskModel(RiskModel):
    """Factor-based risk model: Sigma = X F X' + D^2.

    Computes the covariance matrix from factor loadings (X), a factor
    covariance matrix (F), and a diagonal matrix of idiosyncratic
    volatilities (D).
    """

    def __init__(
        self,
        factor_loadings: FactorLoadings,
        factor_covariances: FactorCovariances,
        idio_vol: IdioVol,
    ):
        self.factor_loadings = factor_loadings
        self.factor_covariances = factor_covariances
        self.idio_vol = idio_vol
        self._tickers = sorted(
            set(factor_loadings["ticker"]) & set(idio_vol["ticker"])
        )

    def _build_factor_loadings_matrix(self, factor_loadings: FactorLoadings) -> np.ndarray:
        """Pivot factor loadings into an (n_assets x n_factors) numpy matrix."""
        return (
            factor_loadings.sort("ticker", "factor")
            .pivot(index="ticker", on="factor", values="loading")
            .drop("ticker")
            .to_numpy()
        )

    def _build_factor_covariance_matrix(self, factor_covariances: FactorCovariances) -> np.ndarray:
        """Pivot factor covariances into a symmetric (n_factors x n_factors) numpy matrix."""
        return (
            factor_covariances.sort("factor_1", "factor_2")
            .pivot(index="factor_1", on="factor_2", values="covariance")
            .drop("factor_1")
            .to_numpy()
        )

    def _build_idio_vol_matrix(self, idio_vol: IdioVol) -> np.ndarray:
        """Build a diagonal matrix of idiosyncratic volatilities."""
        return np.diag(
            idio_vol.sort("ticker")["idio_vol"]
            .to_numpy()
        )

    def build_covariance_matrix(self) -> np.ndarray:
        """Compute the full covariance matrix as X @ F @ X.T + D^2."""
        factor_loadings = self.factor_loadings.filter(pl.col("ticker").is_in(self.tickers))
        idio_vol = self.idio_vol.filter(pl.col("ticker").is_in(self.tickers))

        X = self._build_factor_loadings_matrix(factor_loadings)
        F = self._build_factor_covariance_matrix(self.factor_covariances)
        D = self._build_idio_vol_matrix(idio_vol)

        return X @ F @ X.T + D ** 2
    
    @property
    def tickers(self) -> list[str]:
        return self._tickers 

class FactorRiskModelConstructor(RiskModelConstructor):
    def __init__(
        self,
        factor_loadings: FactorLoadingsProvider,
        factor_covariances: FactorCovariancesProvider,
        idio_vol: IdioVolProvider,
    ):
        self.factor_loadings = factor_loadings
        self.factor_covariances = factor_covariances
        self.idio_vol = idio_vol

    def get_risk_model(self, date_: dt.date) -> FactorRiskModel:
        return FactorRiskModel(
            factor_loadings=self.factor_loadings.get(date_),
            factor_covariances=self.factor_covariances.get(date_),
            idio_vol=self.idio_vol.get(date_)
        )