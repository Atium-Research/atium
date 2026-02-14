from test_data import MyDataAdapter
from harbinger.backtester import Backtester
from harbinger.objectives import MaxUtility
from harbinger.optimizer_constraints import LongOnly, FullyInvested
from harbinger.trading_constraints import MinPositionSize
from harbinger.risk_model import FactorRiskModel
import datetime as dt
import polars as pl

start = dt.date(2023, 1, 1)
end = dt.date(2023, 12, 31)
data = MyDataAdapter(start, end)

bt = Backtester()

results = bt.run(
    start=start,
    end=end,
    data=data,
    initial_capital=100_000,
    objective=MaxUtility(lambda_=100),
    optimizer_constraints=[LongOnly(), FullyInvested()],
    trading_constraints=[MinPositionSize(dollars=1)],
    risk_model=FactorRiskModel(data)
)

print(results)

results_agg = (
    results
    .group_by('date')
    .agg(
        pl.col('value').add(pl.col('pnl')).sum(),
        pl.col('return').mul(pl.col('weight')).sum()
    )
)

print(results_agg)

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(results_agg, x='date', y='value')
plt.savefig('test.png')

print(
    results
    .select(
        pl.col('return').mean().mul(252 * 100).alias('mean_return'),
        pl.col('return').std().mul(pl.lit(252).sqrt() * 100).alias('volatility')
    )
    .with_columns(
        pl.col('mean_return').truediv('volatility').alias('sharpe')
    )
)