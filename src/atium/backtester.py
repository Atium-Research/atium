import datetime as dt
from typing import Literal

import polars as pl
from tqdm import tqdm

from atium.costs import CostModel, NoCost
from atium.data import (BenchmarkWeightsProvider, CalendarProvider,
                        ReturnsProvider)
from atium.result import BacktestResult
from atium.schemas import (BenchmarkReturnsSchema, PortfolioWeightsSchema,
                           PositionResultsSchema)
from atium.strategy import Strategy
from atium.trade_generator import TradeGenerator
from atium.types import PortfolioWeights

RebalanceFrequency = Literal['daily', 'weekly', 'monthly']


class Backtester:
    """Engine that runs a strategy over historical data and tracks portfolio performance.

    On each trading date the backtester generates new weights, deducts
    transaction costs, computes PnL from forward returns, and rolls
    capital forward.
    """

    def _is_rebalance_date(
        self,
        date_: dt.date,
        prev_date: dt.date | None,
        frequency: RebalanceFrequency,
    ) -> bool:
        """Return True if the portfolio should be rebalanced on this date."""
        if prev_date is None:
            return True
        if frequency == 'daily':
            return True
        if frequency == 'weekly':
            y1, w1, _ = prev_date.isocalendar()
            y2, w2, _ = date_.isocalendar()
            return (y1, w1) != (y2, w2)
        if frequency == 'monthly':
            return (prev_date.year, prev_date.month) != (date_.year, date_.month)
        return True

    def run(
        self,
        calendar: CalendarProvider,
        returns: ReturnsProvider,
        strategy: Strategy,
        start: dt.date,
        end: dt.date,
        initial_capital: float,
        cost_model: CostModel | None = None,
        rebalance_frequency: RebalanceFrequency = 'daily',
        benchmark: BenchmarkWeightsProvider | None = None,
        trade_generator: TradeGenerator | None = None,
    ) -> BacktestResult:
        """Execute the backtest and return a BacktestResult.

        Args:
            calendar: Provider for trading dates.
            returns: Provider for next-period forward returns.
            strategy: Strategy that generates portfolio weights each period.
            start: First date of the backtest (inclusive).
            end: Last date of the backtest (inclusive).
            initial_capital: Starting portfolio value in dollars.
            cost_model: Transaction cost model (defaults to NoCost).
            rebalance_frequency: How often to rebalance — 'daily', 'weekly'
                (first trading day of each ISO week), or 'monthly' (first
                trading day of each calendar month).
            benchmark: Optional benchmark provider for benchmark returns.
                When supplied, BacktestResult will include benchmark-relative
                analytics.
            trade_generator: Optional trade generator that applies trading
                constraints to weights on rebalance dates.
        """
        if cost_model is None:
            cost_model = NoCost()

        capital = initial_capital
        holdings: PortfolioWeights | None = None
        prev_date: dt.date | None = None
        results_list: list[pl.DataFrame] = []
        benchmark_returns_list: list[dict] = []

        for date_ in tqdm(calendar.get(start, end), "RUNNING BACKTEST"):
            rebalance = self._is_rebalance_date(date_, prev_date, rebalance_frequency)

            if rebalance:
                new_weights = strategy.generate_weights(date_)
                if trade_generator is not None:
                    new_weights = trade_generator.apply(new_weights, capital)
                costs = cost_model.compute_costs(holdings, new_weights, capital)
                capital -= costs
            else:
                new_weights = holdings.with_columns(pl.lit(date_).alias('date'))

            forward_returns = returns.get(date_)

            results = (
                new_weights
                .join(forward_returns, on=['date', 'ticker'], how='left')
                .with_columns(pl.col('return').fill_null(0))
                .with_columns(
                    pl.col('weight').mul(pl.lit(capital)).alias('value'),
                )
                .with_columns(
                    pl.col('value').mul('return').alias('pnl'),
                )
            )

            invested = results['value'].sum()
            cash = capital - invested
            capital = invested + results['pnl'].sum() + cash

            # Compute drifted weights for next day's holdings
            holdings = PortfolioWeightsSchema.validate(
                results
                .with_columns(
                    ((pl.col('value') + pl.col('pnl')) / capital).alias('weight'),
                )
                .select('date', 'ticker', 'weight')
            )

            if benchmark is not None:
                bm_weights = benchmark.get(date_)
                bm_return = (
                    bm_weights
                    .join(forward_returns, on=['date', 'ticker'], how='left')
                    .with_columns(pl.col('return').fill_null(0))
                    .select(pl.col('weight').mul(pl.col('return')).sum())
                    .item()
                )
                benchmark_returns_list.append({
                    'date': date_,
                    'benchmark_return': float(bm_return),
                })

            prev_date = date_
            results_list.append(results)

        benchmark_returns = (
            BenchmarkReturnsSchema.validate(pl.DataFrame(benchmark_returns_list))
            if benchmark_returns_list
            else None
        )

        return BacktestResult(PositionResultsSchema.validate(pl.concat(results_list)), benchmark_returns)
