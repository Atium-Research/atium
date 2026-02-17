"""Full backtest with weekly rebalancing and transaction costs."""
from demo_data import (
    get_bear_lake_client,
    MyCalendarProvider,
    MyBenchmarkWeightsProvider,
    MyAlphaProvider,
    MyFactorCovariancesProvider,
    MyFactorLoadingsProvider,
    MyIdioVolProvider,
    MyReturnsProvider
)
from atium.risk_model import FactorRiskModel
from atium.optimizer import MVO
from atium.objectives import MaxUtilityWithTargetActiveRisk
from atium.optimizer_constraints import LongOnly, FullyInvested
from atium.trade_generator import TradeGenerator
from atium.trading_constraints import MaxPositionCount, MinPositionSize
from atium.backtester import Backtester
from atium.strategy import OptimizationStrategy
from atium.costs import LinearCost
import datetime as dt

# Parameters
db = get_bear_lake_client()
start = dt.date(2026, 1, 2)
end = dt.date(2026, 2, 13)

# Data providers
calendar_provider = MyCalendarProvider(db, start, end)
alphas_provider = MyAlphaProvider(db, start, end)
factor_loadings_provider = MyFactorLoadingsProvider(db, start, end)
factor_covariances_provider = MyFactorCovariancesProvider(db, start, end)
idio_vol_provider = MyIdioVolProvider(db, start, end)
benchmark_provider = MyBenchmarkWeightsProvider(db, start, end)
returns_provider = MyReturnsProvider(db, start, end)


# Define risk model
risk_model = FactorRiskModel(
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
    alphas=alphas_provider,
    risk_model=risk_model,
    optimizer=optimizer,
    benchmark=benchmark_provider
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
    calendar=calendar_provider,
    returns=returns_provider,
    strategy=strategy,
    start=start,
    end=end,
    initial_capital=100_000,
    cost_model=cost_model,
    rebalance_frequency='weekly',
    benchmark=benchmark_provider,
    trade_generator=trade_generator
)

print(results.summary())
results.plot_equity_curve('test.png')