from test_data import MyDataAdapter
from harbinger import (
    Backtester,
    OptimizationStrategy,
    MaxUtilityWithTargetActiveRisk,
    LongOnly,
    FullyInvested,
    MinPositionSize,
    FactorRiskModel,
    LinearCost,
)
import datetime as dt

start = dt.date(2023, 1, 1)
end = dt.date(2023, 12, 31)
data = MyDataAdapter(start, end)

strategy = OptimizationStrategy(
    alpha_provider=data,
    risk_model=FactorRiskModel(data),
    objective=MaxUtilityWithTargetActiveRisk(target_active_risk=0.05),
    optimizer_constraints=[LongOnly(), FullyInvested()],
    trading_constraints=[MinPositionSize(dollars=1)],
    benchmark_provider=data,
)

bt = Backtester()
result = bt.run(
    market_data=data,
    strategy=strategy,
    cost_model=LinearCost(bps=5),
    start=start,
    end=end,
    initial_capital=100_000,
)

print(result.summary())
result.plot_equity_curve('test.png')
