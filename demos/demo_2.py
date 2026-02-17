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
from atium.risk_model import FactorRiskModelConstructor
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
    alpha_provider=alphas_provider,
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
    rebalance_frequency='weekly',
    initial_capital=100_000,
    calendar=calendar_provider,
    returns=returns_provider,
    benchmark=benchmark_provider,
    strategy=strategy,
    cost_model=cost_model,
    trade_generator=trade_generator
)

print(results.summary())
results.plot_equity_curve('test.png')