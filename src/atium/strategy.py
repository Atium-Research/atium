from abc import ABC, abstractmethod
import datetime as dt
import polars as pl
import numpy as np
from atium.data import AlphaProvider, BenchmarkProvider
from atium.optimizer import MVO
from atium.risk_model import RiskModel


class Strategy(ABC):
    """Base class for portfolio construction strategies."""

    @abstractmethod
    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, weight]."""
        pass


class OptimizationStrategy(Strategy):
    """Portfolio strategy that uses convex optimization to select weights.

    Delegates the actual solve to an Optimizer instance and fetches the
    data it needs (alphas, covariance matrix, benchmark weights).
    """

    def __init__(
        self,
        alphas: AlphaProvider,
        risk_model: RiskModel,
        optimizer: MVO,
        benchmark: BenchmarkProvider | None = None,
    ):
        self.alphas = alphas
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.benchmark = benchmark

    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Generate optimized portfolio weights for the given date and capital."""
        alphas = self.alphas.get(date_)
        tickers = alphas['ticker'].unique().sort().to_list()
        covariance_matrix = self.risk_model.build_covariance_matrix(date_, tickers)

        benchmark_weights = None
        if self.benchmark is not None:
            bm = self.benchmark.get(date_)
            benchmark_weights = (
                pl.DataFrame({'ticker': tickers})
                .join(bm.select('ticker', 'weight'), on='ticker', how='left')
                .with_columns(pl.col('weight').fill_null(0.0))
                .sort('ticker')['weight']
                .to_numpy()
            )

        return self.optimizer.optimize(date_, alphas, covariance_matrix, benchmark_weights)
