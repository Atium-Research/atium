import datetime as dt
import cvxpy as cp
import numpy as np
import polars as pl
from atium.objectives import Objective
from atium.optimizer_constraints import OptimizerConstraint
from atium.types import Alphas, BenchmarkWeights


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
    ) -> pl.DataFrame:
        """Solve the optimization problem and return weights.

        Returns zero weights for all assets if the solver fails.

        Args:
            date_: The date for this optimization.
            alphas: Alpha signals aligned to the asset universe.
            covariance_matrix: n_assets x n_assets covariance matrix.
            benchmark_weights: Optional benchmark weights aligned to the asset universe.
        """
        tickers = alphas.tickers
        alphas_np = alphas.to_numpy()
        n_assets = len(tickers)

        weights = cp.Variable(n_assets)

        build_kwargs = dict(alphas=alphas_np, covariance_matrix=covariance_matrix)
        if benchmark_weights is not None:
            build_kwargs['benchmark_weights'] = benchmark_weights.to_numpy()

        objective = self.objective.build(weights, **build_kwargs)
        constraints = [c.build(weights) for c in self.constraints]

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
