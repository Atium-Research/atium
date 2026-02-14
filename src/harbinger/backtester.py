from harbinger.data import DataAdapter
import polars as pl
from harbinger.objectives import Objective
from harbinger.optimizer_constraints import OptimizerConstraint
from harbinger.trading_constraints import TradingConstraint
import datetime as dt
from harbinger.risk_model import RiskModel
import cvxpy as cp

class Backtester:
    def __init__(self, data: DataAdapter) -> None:
        self.data = data

    def _optimize_portfolio(
        self,
        alphas: pl.DataFrame,
        covariance_matrix: pl.DataFrame,
        objective: Objective, 
        optimizer_constraints: list[OptimizerConstraint]
    ) -> pl.DataFrame:
        alphas_np = alphas.sort('ticker')['alpha'].to_numpy()
        covariance_matrix_np = covariance_matrix.sort('ticker').drop('ticker').to_numpy()

        n_assets = len(alphas)
        weights = cp.Variable(n_assets)
        objective_function = objective.build(weights, alphas=alphas_np, covariance_matrix=covariance_matrix_np)
        constraints = [c.build(weights) for c in optimizer_constraints]

        problem = cp.Problem(objective_function, constraints)
        problem.solve()

        return weights.value
    
    def run(
        self,
        start: dt.date,
        end: dt.date,
        capital: float,
        objective: Objective, 
        optimizer_constraints: list[OptimizerConstraint], 
        trading_constraints: list[TradingConstraint],
        risk_model: RiskModel,
    ) -> pl.DataFrame:
        for date_ in self.data.get_calendar(start, end):
            alphas = self.data.get_alphas(date_)
            tickers = alphas['ticker'].unique().sort().to_list()
            covariance_matrix = risk_model.build_covariance_matrix(date_, tickers)

            weights = self._optimize_portfolio(
                alphas=alphas, 
                covariance_matrix=covariance_matrix, 
                objective=objective, 
                optimizer_constraints=optimizer_constraints
            )

            print(weights)
            break