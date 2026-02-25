import datetime as dt

import bear_lake as bl
import polars as pl
from demo_data import get_bear_lake_client

from atium.factor_model import (estimate_factor_covariances,
                                estimate_factor_model)

db = get_bear_lake_client()
start = dt.date(2020, 7, 27)
end = dt.date(2026, 1, 31)

stock_returns = (
    db.query(
        bl.table('stock_returns')
        .filter(pl.col('date').is_between(start, end))
        .drop('year')
        .sort('date', 'ticker')
    )
)

factor_returns = (
    db.query(
        bl.table('etf_returns')
        .filter(pl.col('date').is_between(start, end))
        .drop('year')
        .sort('date', 'ticker')
    )
)


factor_loadings, idio_vol = estimate_factor_model(
    stock_returns=stock_returns,
    factor_returns=factor_returns,
    window=252,
    half_life=60
)

factor_covariances = estimate_factor_covariances(
    factor_returns=factor_returns,
    window=252,
    half_life=60
)

print(factor_loadings)
print(idio_vol)
print(factor_covariances)