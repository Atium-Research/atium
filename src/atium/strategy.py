from abc import ABC, abstractmethod
import datetime as dt
import polars as pl
from atium.data import AlphaProvider, BenchmarkWeightsProvider
from atium.optimizer import MVO
from atium.risk_model import RiskModel
from atium.types import Alphas, BenchmarkWeights


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
        benchmark: BenchmarkWeightsProvider | None = None,
    ):
        self.alphas = alphas
        self.risk_model = risk_model
        self.optimizer = optimizer
        self.benchmark = benchmark

    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Generate optimized portfolio weights for the given date and capital."""
        alphas = Alphas(self.alphas.get(date_))
        tickers = alphas.tickers
        covariance_matrix = self.risk_model.build_covariance_matrix(date_, tickers)

        benchmark_weights = None
        if self.benchmark is not None:
            benchmark_weights = BenchmarkWeights(self.benchmark.get(date_)).align_to(tickers)

        return self.optimizer.optimize(date_, alphas, covariance_matrix, benchmark_weights)
