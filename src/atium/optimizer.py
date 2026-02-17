import datetime as dt
import cvxpy as cp
import numpy as np
import polars as pl
from atium.objectives import Objective
from atium.optimizer_constraints import OptimizerConstraint
from atium.models import Alphas, BenchmarkWeights, PortfolioWeights


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
        covariance_matrix: np.ndarray,
        benchmark_weights: BenchmarkWeights | None = None,
    ) -> PortfolioWeights:
        tickers = alphas['ticker'].sort().to_list()
        alphas_np = alphas['alpha'].to_numpy()
        n_assets = len(tickers)

        weights = cp.Variable(n_assets)

        build_kwargs = dict(alphas=alphas_np, covariance_matrix=covariance_matrix)
        if benchmark_weights is not None:
            build_kwargs['benchmark_weights'] = benchmark_weights.sort('ticker')['weight'].to_numpy()

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

        return PortfolioWeights(pl.DataFrame({
            'date': [date_] * n_assets,
            'ticker': tickers,
            'weight': weights.value,
        }))
