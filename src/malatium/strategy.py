from abc import ABC, abstractmethod
import datetime as dt
import polars as pl
import cvxpy as cp
import numpy as np
from malatium.data import AlphaProvider, BenchmarkProvider
from malatium.objectives import Objective
from malatium.optimizer_constraints import OptimizerConstraint
from malatium.trading_constraints import TradingConstraint
from malatium.risk_model import RiskModel


class Strategy(ABC):
    """Base class for portfolio construction strategies."""

    @abstractmethod
    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, weight]."""
        pass


class OptimizationStrategy(Strategy):
    """Portfolio strategy that uses convex optimization to select weights.

    Builds a CVXPY problem from an objective function and a set of constraints,
    solves it, and then applies post-optimization trading constraints.
    """

    def __init__(
        self,
        alphas: AlphaProvider,
        risk_model: RiskModel,
        objective: Objective,
        optimizer_constraints: list[OptimizerConstraint],
        trading_constraints: list[TradingConstraint],
        benchmark: BenchmarkProvider | None = None,
    ):
        self.alphas = alphas
        self.risk_model = risk_model
        self.objective = objective
        self.optimizer_constraints = optimizer_constraints
        self.trading_constraints = trading_constraints
        self.benchmark = benchmark

    def _optimize(
        self,
        date_: dt.date,
        alphas: pl.DataFrame,
        covariance_matrix: np.ndarray,
    ) -> pl.DataFrame:
        """Solve the convex optimization problem and return raw weights.

        Returns zero weights for all assets if the solver fails to find
        an optimal solution.
        """
        tickers = alphas['ticker'].unique().sort().to_list()
        alphas_np = alphas.sort('ticker')['alpha'].to_numpy()
        n_assets = len(tickers)

        weights = cp.Variable(n_assets)

        build_kwargs = dict(alphas=alphas_np, covariance_matrix=covariance_matrix)
        if self.benchmark is not None:
            bm = self.benchmark.get(date_)
            bm_weights = (
                pl.DataFrame({'ticker': tickers})
                .join(bm.select('ticker', 'weight'), on='ticker', how='left')
                .with_columns(pl.col('weight').fill_null(0.0))
                .sort('ticker')['weight']
                .to_numpy()
            )
            build_kwargs['benchmark_weights'] = bm_weights

        objective = self.objective.build(weights, **build_kwargs)
        constraints = [c.build(weights) for c in self.optimizer_constraints]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ('optimal', 'optimal_inaccurate'):
            return pl.DataFrame({
                'date': [date_] * n_assets,
                'ticker': tickers,
                'weight': [0.0] * n_assets,
            })

        return pl.DataFrame({
            'date': [date_] * n_assets,
            'ticker': tickers,
            'weight': weights.value,
        })

    def _apply_trading_constraints(
        self, weights: pl.DataFrame, capital: float
    ) -> pl.DataFrame:
        """Apply each trading constraint to the weights sequentially."""
        for constraint in self.trading_constraints:
            weights = constraint.apply(weights, capital=capital)
        return weights

    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Generate optimized portfolio weights for the given date and capital."""
        alphas = self.alphas.get(date_)
        tickers = alphas['ticker'].unique().sort().to_list()
        covariance_matrix = self.risk_model.build_covariance_matrix(date_, tickers)

        weights = self._optimize(date_, alphas, covariance_matrix)
        weights = self._apply_trading_constraints(weights, capital)

        return weights
