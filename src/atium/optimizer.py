import datetime as dt

import cvxpy as cp
import polars as pl

from atium.objectives import Objective
from atium.optimizer_constraints import OptimizerConstraint
from atium.risk_model import RiskModel
from atium.schemas import PortfolioWeightsSchema
from atium.types import Alphas, BenchmarkWeights, Betas, PortfolioWeights


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
        risk_model: RiskModel,
        alphas: Alphas | None = None,
        benchmark_weights: BenchmarkWeights | None = None,
        betas: Betas | None = None,
    ) -> PortfolioWeights:
        # Get covariance matrix
        covariance_matrix = risk_model.build_covariance_matrix()

        # Get unique tickers from risk model
        tickers = risk_model.tickers

        # Filter alphas
        if alphas is not None:
            alphas_np = (
                pl.DataFrame({'ticker': tickers})
                .join(alphas.select(['ticker', 'alpha']), on='ticker', how='left')
                .fill_null(0.0)
                .sort('ticker')['alpha']
                .to_numpy()
            )

        # Filter benchmark weights
        if benchmark_weights is not None:
            benchmark_weights_np = benchmark_weights.filter(pl.col('ticker').is_in(tickers)).sort('ticker')['weight'].to_numpy()

        # Filter betas
        if betas is not None:
            betas_aligned = (
                pl.DataFrame({'ticker': tickers})
                .join(betas.select(['ticker', 'beta']), on='ticker', how='left')
                .sort('ticker')
            )
            missing = betas_aligned.filter(pl.col('beta').is_null())['ticker'].to_list()
            if missing:
                raise ValueError(f"BetaProvider is missing betas for tickers: {missing}")
            betas_np = betas_aligned['beta'].to_numpy()

        # Create weights variable
        n_assets = len(tickers)
        weights = cp.Variable(n_assets)

        # Build objective kwargs
        build_kwargs = dict(covariance_matrix=covariance_matrix, constraints=self.constraints)
        if alphas is not None:
            build_kwargs['alphas'] = alphas_np
        if benchmark_weights is not None:
            build_kwargs['benchmark_weights'] = benchmark_weights_np
        if betas is not None:
            build_kwargs['betas'] = betas_np

        # Define problem and solve
        objective = self.objective.build(weights, **build_kwargs)
        constraints = [c.build(weights, **build_kwargs) for c in self.constraints]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ('optimal', 'optimal_inaccurate'):
            return PortfolioWeightsSchema.validate(pl.DataFrame({
                'date': [date_] * n_assets,
                'ticker': tickers,
                'weight': [0.0] * n_assets,
            }))

        return PortfolioWeightsSchema.validate(pl.DataFrame({
            'date': [date_] * n_assets,
            'ticker': tickers,
            'weight': weights.value,
        }))
