# Malatium

Quant primitives for building a hedge fund. Malatium provides a modular backtesting framework with convex portfolio optimization, pluggable risk models, transaction cost modeling, and flexible data providers.

## Installation

```bash
pip install malatium
```

## Quick Start

```python
from malatium.backtester import Backtester
from malatium.strategy import OptimizationStrategy
from malatium.objectives import MaxUtilityWithTargetActiveRisk
from malatium.optimizer_constraints import LongOnly, FullyInvested
from malatium.trading_constraints import MinPositionSize
from malatium.risk_model import FactorRiskModel
from malatium.costs import LinearCost
import datetime as dt

# 1. Create your data providers (see "Data Providers" below)
start = dt.date(2026, 1, 1)
end = dt.date(2026, 12, 31)

calendar = MyCalendarProvider(start, end)
returns = MyReturnsProvider(start, end)
alphas = MyAlphaProvider(start, end)
factor_loadings = MyFactorLoadingsProvider(start, end)
factor_covariances = MyFactorCovariancesProvider(start, end)
idio_vol = MyIdioVolProvider(start, end)
benchmark = MyBenchmarkProvider(start, end)

# 2. Define a strategy
strategy = OptimizationStrategy(
    alphas=alphas,
    risk_model=FactorRiskModel(factor_loadings, factor_covariances, idio_vol),
    objective=MaxUtilityWithTargetActiveRisk(target_active_risk=0.05),
    optimizer_constraints=[LongOnly(), FullyInvested()],
    trading_constraints=[MinPositionSize(dollars=1)],
    benchmark=benchmark,
)

# 3. Run the backtest
bt = Backtester()
result = bt.run(
    calendar=calendar,
    returns=returns,
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

## Data Providers

Allomancy uses Protocol-based data providers — one per dataset. Each provider has a single `get()` method. Any class with a matching `get()` signature satisfies the protocol automatically (no subclassing required).

| Provider | `get()` signature | Returns |
|----------|-------------------|---------|
| `CalendarProvider` | `get(start, end) -> list[dt.date]` | Trading dates in range |
| `ReturnsProvider` | `get(date) -> DataFrame` | `[date, ticker, return]` |
| `AlphaProvider` | `get(date) -> DataFrame` | `[date, ticker, alpha]` |
| `FactorLoadingsProvider` | `get(date) -> DataFrame` | `[date, ticker, factor, loading]` |
| `FactorCovariancesProvider` | `get(date) -> DataFrame` | `[date, factor_1, factor_2, covariance]` |
| `IdioVolProvider` | `get(date) -> DataFrame` | `[date, ticker, idio_vol]` |
| `BenchmarkProvider` | `get(date) -> DataFrame` | `[date, ticker, weight]` |

### Writing a Data Provider

Each provider is a simple class with a `get()` method. Pre-loading data into memory during `__init__` is recommended for backtest speed.

```python
import datetime as dt
import polars as pl


class MyReturnsProvider:
    def __init__(self, start: dt.date, end: dt.date) -> None:
        self._returns = load_returns_from_db(start, end)

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self._returns.filter(pl.col("date").eq(date_))


class MyAlphaProvider:
    def __init__(self, start: dt.date, end: dt.date) -> None:
        self._alphas = load_alphas_from_db(start, end)

    def get(self, date_: dt.date) -> pl.DataFrame:
        return self._alphas.filter(pl.col("date").eq(date_))
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

- **`FactorRiskModel(factor_loadings, factor_covariances, idio_vol)`** — Computes `Sigma = X F X' + D^2` from factor loadings (`X`), factor covariances (`F`), and idiosyncratic volatilities (`D`).

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
