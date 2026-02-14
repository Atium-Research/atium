# Harbinger

Quant primitives for building a hedge fund. Harbinger provides a modular backtesting framework with convex portfolio optimization, pluggable risk models, transaction cost modeling, and flexible data adapters.

## Installation

```bash
pip install harbinger
```

## Quick Start

```python
from harbinger.backtester import Backtester
from harbinger.strategy import OptimizationStrategy
from harbinger.objectives import MaxUtilityWithTargetActiveRisk
from harbinger.optimizer_constraints import LongOnly, FullyInvested
from harbinger.trading_constraints import MinPositionSize
from harbinger.risk_model import FactorRiskModel
from harbinger.costs import LinearCost
import datetime as dt

# 1. Create your data adapter (see "Data Adapters" below)
start = dt.date(2026, 1, 1)
end = dt.date(2026, 12, 31)
data = MyDataAdapter(start, end)

# 2. Define a strategy
strategy = OptimizationStrategy(
    alpha_provider=data,
    risk_model=FactorRiskModel(data),
    objective=MaxUtilityWithTargetActiveRisk(target_active_risk=0.05),
    optimizer_constraints=[LongOnly(), FullyInvested()],
    trading_constraints=[MinPositionSize(dollars=1)],
    benchmark_provider=data,
)

# 3. Run the backtest
bt = Backtester()
result = bt.run(
    market_data=data,
    strategy=strategy,
    cost_model=LinearCost(bps=5),
    start=start,
    end=end,
    initial_capital=100_000,
)

# 4. Analyze results
print(result.summary())
result.plot_equity_curve("equity_curve.png")
```

## Data Adapters

Harbinger uses three abstract data provider interfaces that you implement to connect your own data:

| Interface | Purpose |
|-----------|---------|
| `MarketDataProvider` | Trading calendar, universe, prices, forward returns |
| `AlphaProvider` | Signals, scores, and alpha (expected return) estimates |
| `RiskDataProvider` | Benchmark weights, betas, factor loadings, factor covariances, idiosyncratic volatility |

The `DataAdapter` class inherits from all three, so you can implement a single class that satisfies every provider role.

### Writing a Data Adapter

Subclass `DataAdapter` and implement every abstract method. Each method receives a date and returns a Polars DataFrame (or a list for calendar/universe methods).

```python
import datetime as dt
import polars as pl
from harbinger.data import DataAdapter


class MyDataAdapter(DataAdapter):
    def __init__(self, start: dt.date, end: dt.date) -> None:
        # Load or connect to your data source here.
        # Pre-loading into memory is recommended for backtest speed.
        self._prices = self._load_prices(start, end)
        # ... load other tables ...

    # -- MarketDataProvider --

    def get_calendar(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """Return the list of trading dates in the range."""
        ...

    def get_universe(self, date_: dt.date) -> list[str]:
        """Return ticker symbols available on this date."""
        ...

    def get_prices(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, price]."""
        ...

    def get_forward_returns(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, return]."""
        ...

    # -- AlphaProvider --

    def get_signals(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, signal]."""
        ...

    def get_scores(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, score]."""
        ...

    def get_alphas(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, alpha]."""
        ...

    # -- RiskDataProvider --

    def get_benchmark_weights(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, weight]."""
        ...

    def get_betas(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, beta]."""
        ...

    def get_factor_loadings(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, factor, loading]."""
        ...

    def get_factor_covariances(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, factor_1, factor_2, covariance]."""
        ...

    def get_idio_vol(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, idio_vol]."""
        ...
```

### Pre-loading Pattern

For best performance, load all data into memory during `__init__` and filter by date in each getter:

```python
def __init__(self, start: dt.date, end: dt.date) -> None:
    self._prices = load_prices_from_db(start, end)

def get_prices(self, date_: dt.date) -> pl.DataFrame:
    return self._prices.filter(pl.col("date").eq(date_))
```

## Components

### Objectives

Control how the optimizer selects portfolio weights.

- **`MaxUtility(lambda_)`** — Mean-variance optimization: maximize `w @ alpha - lambda/2 * w' Sigma w`.
- **`MaxUtilityWithTargetActiveRisk(target_active_risk)`** — Iteratively finds the risk-aversion parameter that produces a portfolio matching the target active risk (annualized tracking error vs. benchmark).

### Optimizer Constraints

Hard constraints passed to the CVXPY solver.

- **`LongOnly()`** — No short positions (`w >= 0`).
- **`FullyInvested()`** — Weights sum to one (`sum(w) == 1`).

Implement `OptimizerConstraint` to add your own.

### Trading Constraints

Post-optimization filters applied to the resulting weights.

- **`MinPositionSize(dollars)`** — Zeroes out any position smaller than the given dollar amount.

Implement `TradingConstraint` to add your own.

### Risk Models

Build the covariance matrix used by the optimizer.

- **`FactorRiskModel(risk_data_provider)`** — Computes `Sigma = X F X' + D^2` from factor loadings (`X`), factor covariances (`F`), and idiosyncratic volatilities (`D`).

Implement `RiskModel` to add your own.

### Cost Models

Estimate transaction costs deducted from capital at each rebalance.

- **`NoCost()`** — Zero cost.
- **`LinearCost(bps)`** — Cost proportional to turnover: `turnover * capital * bps / 10,000`.

Implement `CostModel` to add your own.

### Backtest Results

`BacktestResult` provides:

| Method | Returns |
|--------|---------|
| `summary()` | DataFrame with annualized return, volatility, Sharpe ratio, and max drawdown |
| `portfolio_returns()` | Daily portfolio returns and values |
| `sharpe_ratio()` | Annualized Sharpe ratio |
| `annualized_return()` | Annualized return (%) |
| `annualized_volatility()` | Annualized volatility (%) |
| `max_drawdown()` | Maximum peak-to-trough drawdown |
| `plot_equity_curve(path)` | Save an equity curve chart to disk |
