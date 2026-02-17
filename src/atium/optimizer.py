import datetime as dt
import cvxpy as cp
import numpy as np
import polars as pl
from atium.objectives import Objective
from atium.optimizer_constraints import OptimizerConstraint
from atium.models import Alphas, BenchmarkWeights, PortfolioWeights
from atium.risk_model import RiskModel


class MVO:
    """Mean-Variance Optimizer using CVXPY.

    Builds a convex optimization problem from an objective and a set of
    constraints, then solves it to produce portfolio weights.

    Args:
        objective: Objective function to maximize (e.g. MaxUtility).
        constraints: Hard constraints passed to the CVXPY solver.
    """

    def __init__(
        self,
        objective: Objective,
        constraints: list[OptimizerConstraint],
    ):
        self.objective = objective
        self.constraints = constraints

    def optimize(
        self,
        date_: dt.date,
        alphas: Alphas,
        risk_model: RiskModel,
        benchmark_weights: BenchmarkWeights | None = None,
    ) -> PortfolioWeights:
        # Get covariance matrix
        covariance_matrix = risk_model.build_covariance_matrix()

        # Get unique tickers from risk model
        tickers = risk_model.tickers

        # Filter alphas
        alphas_np = alphas.filter(pl.col('ticker').is_in(tickers)).sort('ticker')['alpha'].to_numpy()

        # Filter benchmark weights
        if benchmark_weights is not None:
            benchmark_weights_np = benchmark_weights.filter(pl.col('ticker').is_in(tickers)).sort('ticker')['weight'].to_numpy()

        # Create weights variable
        n_assets = len(tickers)
        weights = cp.Variable(n_assets)

        # Build objective kwargs
        build_kwargs = dict(alphas=alphas_np, covariance_matrix=covariance_matrix)
        if benchmark_weights is not None:
            build_kwargs['benchmark_weights'] = benchmark_weights_np

        # Define problem and solve
        objective = self.objective.build(weights, **build_kwargs)
        constraints = [c.build(weights) for c in self.constraints]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ('optimal', 'optimal_inaccurate'):
            return PortfolioWeights(pl.DataFrame({
                'date': [date_] * n_assets,
                'ticker': tickers,
                'weight': [0.0] * n_assets,
            }))

        return PortfolioWeights.validate(pl.DataFrame({
            'date': [date_] * n_assets,
            'ticker': tickers,
            'weight': weights.value,
        }))
