from abc import ABC, abstractmethod
import datetime as dt
import polars as pl
import cvxpy as cp
import numpy as np
from harbinger.data import AlphaProvider
from harbinger.objectives import Objective
from harbinger.optimizer_constraints import OptimizerConstraint
from harbinger.trading_constraints import TradingConstraint
from harbinger.risk_model import RiskModel


class Strategy(ABC):
    @abstractmethod
    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, weight]."""
        pass


class OptimizationStrategy(Strategy):
    def __init__(
        self,
        alpha_provider: AlphaProvider,
        risk_model: RiskModel,
        objective: Objective,
        optimizer_constraints: list[OptimizerConstraint],
        trading_constraints: list[TradingConstraint],
    ):
        self.alpha_provider = alpha_provider
        self.risk_model = risk_model
        self.objective = objective
        self.optimizer_constraints = optimizer_constraints
        self.trading_constraints = trading_constraints

    def _optimize(
        self,
        date_: dt.date,
        alphas: pl.DataFrame,
        covariance_matrix: np.ndarray,
    ) -> pl.DataFrame:
        tickers = alphas['ticker'].unique().sort().to_list()
        alphas_np = alphas.sort('ticker')['alpha'].to_numpy()
        n_assets = len(tickers)

        weights = cp.Variable(n_assets)
        objective = self.objective.build(
            weights, alphas=alphas_np, covariance_matrix=covariance_matrix
        )
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
        for constraint in self.trading_constraints:
            weights = constraint.apply(weights, capital=capital)
        return weights

    def generate_weights(self, date_: dt.date, capital: float) -> pl.DataFrame:
        alphas = self.alpha_provider.get_alphas(date_)
        tickers = alphas['ticker'].unique().sort().to_list()
        covariance_matrix = self.risk_model.build_covariance_matrix(date_, tickers)

        weights = self._optimize(date_, alphas, covariance_matrix)
        weights = self._apply_trading_constraints(weights, capital)

        return weights
