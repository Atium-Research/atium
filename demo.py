import os
import bear_lake as bl
from data import (
    MyCalendarProvider, 
    MyBenchmarkProvider, 
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
import datetime as dt
import polars as pl

# Data connection
def get_bear_lake_client():
    access_key_id = os.getenv("ACCESS_KEY_ID")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    region = os.getenv("REGION")
    endpoint = os.getenv("ENDPOINT")
    bucket = os.getenv("BUCKET")

    storage_options = {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
        "region": region,
        "endpoint_url": endpoint,
    }

    url = f"s3://{bucket}"

    return bl.connect_s3(path=url, storage_options=storage_options)

# Parameters
db = get_bear_lake_client()
start = dt.date(2026, 1, 2)
end = dt.date(2026, 2, 13)

# Data providers
alphas_provider = MyAlphaProvider(db, start, end)
factor_loadings_provider = MyFactorLoadingsProvider(db, start, end)
factor_covariances_provider = MyFactorCovariancesProvider(db, start, end)
idio_vol_provider = MyIdioVolProvider(db, start, end)
benchmark_provider = MyBenchmarkProvider(db, start, end)

# Get alphas
alphas = alphas_provider.get(end)

# Get valid tickers
tickers = alphas['ticker'].to_list()

# Get risk model
risk_model = FactorRiskModel(
    factor_loadings=factor_loadings_provider,
    factor_covariances=factor_covariances_provider,
    idio_vol=idio_vol_provider
)

# Get covariance matrix
covariance_matrix = risk_model.build_covariance_matrix(end, tickers)

# Find optimal weights
optimizer = MVO(
    objective=MaxUtilityWithTargetActiveRisk(target_active_risk=.05),
    constraints=[LongOnly(), FullyInvested()]
)
weights = optimizer.optimize(
    date_=end,
    alphas=alphas,
    covariance_matrix=covariance_matrix,
    benchmark_weights=benchmark_provider.get(end).filter(pl.col('ticker').is_in(tickers)).sort('ticker')['weight'].to_numpy()
)

# Apply trading constraints
trade_generator = TradeGenerator(
    constraints=[MinPositionSize(dollars=1), MaxPositionCount(max_positions=10)]
)
constrained_weights = trade_generator.apply(weights=weights, capital=100_000)

# Print results
print(constrained_weights.sort('weight', descending=True).head(10))