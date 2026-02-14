from harbinger.data import MarketDataProvider
from harbinger.strategy import Strategy
from harbinger.costs import CostModel, NoCost
from harbinger.result import BacktestResult
import polars as pl
import datetime as dt
from tqdm import tqdm


class Backtester:
    """Engine that runs a strategy over historical data and tracks portfolio performance.

    On each trading date the backtester generates new weights, deducts
    transaction costs, computes PnL from forward returns, and rolls
    capital forward.
    """

    def run(
        self,
        market_data: MarketDataProvider,
        strategy: Strategy,
        start: dt.date,
        end: dt.date,
        initial_capital: float,
        cost_model: CostModel | None = None,
    ) -> BacktestResult:
        """Execute the backtest and return a BacktestResult.

        Args:
            market_data: Provider for calendar and forward returns.
            strategy: Strategy that generates portfolio weights each period.
            start: First date of the backtest (inclusive).
            end: Last date of the backtest (inclusive).
            initial_capital: Starting portfolio value in dollars.
            cost_model: Transaction cost model (defaults to NoCost).
        """
        if cost_model is None:
            cost_model = NoCost()

        capital = initial_capital
        prev_weights: pl.DataFrame | None = None
        results_list: list[pl.DataFrame] = []

        for date_ in tqdm(market_data.get_calendar(start, end), "RUNNING BACKTEST"):
            new_weights = strategy.generate_weights(date_, capital)

            costs = cost_model.compute_costs(prev_weights, new_weights, capital)
            capital -= costs

            forward_returns = market_data.get_forward_returns(date_)

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

            prev_weights = new_weights
            results_list.append(results)

        return BacktestResult(pl.concat(results_list))
