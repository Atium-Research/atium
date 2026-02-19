"""Full backtest using custom alphas."""
import datetime as dt

import bear_lake as bl
import polars as pl
from demo_data import (MyAlphaProvider, MyBenchmarkWeightsProvider,
                       MyCalendarProvider, MyFactorCovariancesProvider,
                       MyFactorLoadingsProvider, MyIdioVolProvider,
                       MyReturnsProvider, get_bear_lake_client)

from atium.backtester import Backtester
from atium.costs import LinearCost
from atium.objectives import MaxUtilityWithTargetActiveRisk
from atium.optimizer import MVO
from atium.optimizer_constraints import FullyInvested, LongOnly
from atium.risk_model import FactorRiskModelConstructor
from atium.signals import compute_alphas, compute_scores
from atium.strategy import OptimizationStrategy
from atium.trade_generator import TradeGenerator
from atium.trading_constraints import MaxPositionCount, MinPositionSize

# Parameters
db = get_bear_lake_client()
start = dt.date(2026, 1, 2)
end = dt.date(2026, 2, 19)

# Alpha calculation
universe = db.query(bl.table('universe').drop('year').filter(pl.col('date').is_between(start, end)).sort('ticker', 'date'))
idio_vol = db.query(bl.table('idio_vol').drop('year').filter(pl.col('date').is_between(start, end)).sort('ticker', 'date'))
signals = (
    db.query(
        bl.table('stock_returns')
        .drop('year')
        .sort('date', 'ticker')
        .with_columns(
            pl.col('return')
            .log1p()
            .rolling_sum(21)
            .mul(-1)
            .over('ticker')
            .alias('signal'),
        )
        .filter(pl.col('date').is_between(start, end))
        .sort('date', 'ticker')
    )
)
scores = compute_scores(signals)
alphas = compute_alphas(universe, scores, idio_vol)

# Data providers
calendar_provider = MyCalendarProvider(db, start, end)
factor_loadings_provider = MyFactorLoadingsProvider(db, start, end)
factor_covariances_provider = MyFactorCovariancesProvider(db, start, end)
idio_vol_provider = MyIdioVolProvider(db, start, end)
benchmark_provider = MyBenchmarkWeightsProvider(db, start, end)
returns_provider = MyReturnsProvider(db, start, end)


# Define risk model
risk_model_constructor = FactorRiskModelConstructor(
    factor_loadings=factor_loadings_provider,
    factor_covariances=factor_covariances_provider,
    idio_vol=idio_vol_provider
)

# Define optimizer
optimizer = MVO(
    objective=MaxUtilityWithTargetActiveRisk(target_active_risk=.05),
    constraints=[LongOnly(), FullyInvested()]
)

# Define strategy
strategy = OptimizationStrategy(
    alpha_provider=MyAlphaProvider.from_df(alphas),
    benchmark_weights_provider=benchmark_provider,
    risk_model_constructor=risk_model_constructor,
    optimizer=optimizer,
)

# Define transaction cost model
cost_model = LinearCost(bps=5)

# Define trading constraints
trade_generator = TradeGenerator(
    constraints=[MinPositionSize(dollars=1), MaxPositionCount(max_positions=10)]
)

# Run backtest
backtester = Backtester()
results = backtester.run(
    start=start,
    end=end,
    rebalance_frequency='daily',
    initial_capital=100_000,
    calendar_provider=calendar_provider,
    returns_provider=returns_provider,
    strategy=strategy,
    cost_model=cost_model,
    trade_generator=trade_generator
)

print(results)