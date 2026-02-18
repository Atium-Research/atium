import datetime as dt
from abc import ABC, abstractmethod

from atium.data import AlphaProvider, BenchmarkWeightsProvider
from atium.optimizer import MVO
from atium.risk_model import RiskModelConstructor
from atium.types import PortfolioWeights


class Strategy(ABC):
    """Base class for portfolio construction strategies."""

    @abstractmethod
    def generate_weights(self, date_: dt.date) -> PortfolioWeights:
        """Return a DataFrame with columns [date, ticker, weight]."""
        pass


class OptimizationStrategy(Strategy):
    """Portfolio strategy that uses convex optimization to select weights.

    Delegates the actual solve to an Optimizer instance and fetches the
    data it needs (alphas, covariance matrix, benchmark weights).
    """

    def __init__(
        self,
        alpha_provider: AlphaProvider,
        risk_model_constructor: RiskModelConstructor,
        optimizer: MVO,
        benchmark_weights_provider: BenchmarkWeightsProvider | None = None,
    ):
        self.alpha_provider = alpha_provider
        self.risk_model_constructor = risk_model_constructor
        self.optimizer = optimizer
        self.benchmark_weights_provider = benchmark_weights_provider

    def generate_weights(self, date_: dt.date) -> PortfolioWeights:
        """Generate optimized portfolio weights for the given date."""
        alphas = self.alpha_provider.get(date_)
        risk_model = self.risk_model_constructor.get_risk_model(date_)

        benchmark_weights = None
        if self.benchmark_weights_provider is not None:
            benchmark_weights = self.benchmark_weights_provider.get(date_)

        return self.optimizer.optimize(
            date_=date_, 
            alphas=alphas, 
            risk_model=risk_model, 
            benchmark_weights=benchmark_weights
        )
