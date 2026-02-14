from test_data import MyDataAdapter
from harbinger.backtester import Backtester
from harbinger.objectives import MaxUtility
from harbinger.optimizer_constraints import LongOnly, FullyInvested
from harbinger.trading_constraints import MinPositionSize
from harbinger.risk_model import FactorRiskModel
import datetime as dt

start = dt.date(2023, 1, 1)
end = dt.date(2023, 1, 31)
data = MyDataAdapter(start, end)

bt = Backtester(data=data)

print("RUNNING BACKTEST...")
results = bt.run(
    start=start,
    end=end,
    capital=100_000,
    objective=MaxUtility(lambda_=100),
    optimizer_constraints=[LongOnly(), FullyInvested()],
    trading_constraints=[MinPositionSize(dollars=1)],
    risk_model=FactorRiskModel(data)
)

print(results)