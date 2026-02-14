from harbinger.data import DataAdapter
import polars as pl
from harbinger.objectives import Objective
from harbinger.optimizer_constraints import OptimizerConstraint
from harbinger.trading_constraints import TradingConstraint
import datetime as dt
from harbinger.risk_model import RiskModel
import cvxpy as cp
from tqdm import tqdm

class Backtester:
    def _optimize_portfolio(
        self,
        date_: dt.date,
        alphas: pl.DataFrame,
        covariance_matrix: pl.DataFrame,
        objective: Objective, 
        optimizer_constraints: list[OptimizerConstraint]
    ) -> pl.DataFrame:
        tickers = alphas['ticker'].unique().sort().to_list()
        alphas_np = alphas.sort('ticker')['alpha'].to_numpy()
        covariance_matrix_np = covariance_matrix.sort('ticker').drop('ticker').to_numpy()

        n_assets = len(alphas)
        weights = cp.Variable(n_assets)
        objective_function = objective.build(weights, alphas=alphas_np, covariance_matrix=covariance_matrix_np)
        constraints = [c.build(weights) for c in optimizer_constraints]

        problem = cp.Problem(objective_function, constraints)
        problem.solve()

        return pl.DataFrame({
            'date': date_,
            'ticker': tickers,
            'weight': weights.value        
        })
    
    def _apply_trading_constraints(
        self,
        capital: float,
        initial_weights: pl.DataFrame, 
        trading_constraints: list[TradingConstraint]
    ) -> pl.DataFrame:
        weights = initial_weights
        for trading_constraint in trading_constraints:
            weights = trading_constraint.apply(weights, capital=capital)
        return weights
    
    def run(
        self,
        data: DataAdapter,
        start: dt.date,
        end: dt.date,
        initial_capital: float,
        objective: Objective, 
        optimizer_constraints: list[OptimizerConstraint], 
        trading_constraints: list[TradingConstraint],
        risk_model: RiskModel,
    ) -> pl.DataFrame:
        capital = initial_capital
        results_list = []
        for date_ in tqdm(data.get_calendar(start, end), "RUNNING BACKTEST"):
            alphas = data.get_alphas(date_)
            tickers = alphas['ticker'].unique().sort().to_list()
            covariance_matrix = risk_model.build_covariance_matrix(date_, tickers)

            weights = self._optimize_portfolio(
                date_=date_,
                alphas=alphas, 
                covariance_matrix=covariance_matrix, 
                objective=objective, 
                optimizer_constraints=optimizer_constraints
            )

            constrained_weights = self._apply_trading_constraints(
                capital=capital,
                initial_weights=weights,
                trading_constraints=trading_constraints
            )

            forward_returns = data.get_forward_returns(date_)

            results = (
                constrained_weights
                .join(
                    other=forward_returns,
                    on=['date', 'ticker'],
                    how='left'
                )
                .with_columns(
                    pl.col('weight').mul(pl.lit(capital)).alias('value'),
                )
                .with_columns(
                    pl.col('value').mul('return').alias('pnl')
                )
            )

            capital = results['value'].sum() + results['pnl'].sum()
            results_list.append(results)

        return pl.concat(results_list)