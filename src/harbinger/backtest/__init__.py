"""
Harbinger Backtester

Single entry point for running backtests with MVO and trading constraints.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from harbinger.constraints import (
    Constraint,
    MaxWeight,
    FullyInvested,
    MinPositionValue,
    MaxTurnover,
    apply_optimizer_constraints,
    apply_trading_constraints,
)


@dataclass
class BacktestConfig:
    """
    Configuration for backtest.
    
    Risk control modes:
    - Fixed gamma: Set `gamma` to a float value (e.g., 1.0, 10.0, 100.0)
    - Target active risk: Set `gamma="dynamic"` and `target_active_risk` to desired level (e.g., 0.05)
    
    When using dynamic gamma, the optimizer iteratively finds the gamma that achieves
    the target active risk through the MVO optimization itself (not via constraints).
    """
    
    initial_capital: float = 100_000.0
    
    # Risk control: either fixed gamma OR dynamic with target_active_risk
    gamma: float | Literal["dynamic"] = "dynamic"
    target_active_risk: float | None = 0.05  # Only used when gamma="dynamic"
    
    # Constraints (list of Constraint objects)
    # Note: Do NOT include TargetActiveRisk constraint - use target_active_risk param instead
    optimizer_constraints: list[Constraint] = field(default_factory=list)
    trading_constraints: list[Constraint] = field(default_factory=list)
    
    # Costs
    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    
    @classmethod
    def default(cls) -> "BacktestConfig":
        """Create config with sensible defaults using dynamic gamma."""
        return cls(
            gamma="dynamic",
            target_active_risk=0.05,
            optimizer_constraints=[
                MaxWeight(max_weight=0.10),
            ],
            trading_constraints=[
                MinPositionValue(min_value=1.0),
                MaxTurnover(max_turnover=1.0),
            ],
        )
    
    @classmethod
    def with_fixed_gamma(cls, gamma: float = 100.0) -> "BacktestConfig":
        """Create config with fixed gamma (risk aversion)."""
        return cls(
            gamma=gamma,
            target_active_risk=None,
            optimizer_constraints=[
                MaxWeight(max_weight=0.10),
                FullyInvested(),
            ],
            trading_constraints=[
                MinPositionValue(min_value=1.0),
                MaxTurnover(max_turnover=1.0),
            ],
        )


@dataclass
class BacktestResult:
    """Results from a backtest."""
    
    daily_results: pl.DataFrame  # date, portfolio_value, pnl, return, turnover
    weights: pl.DataFrame        # date, ticker, weight
    trades: pl.DataFrame         # date, ticker, shares, value, cost
    
    def summary(self) -> dict:
        """Calculate summary statistics."""
        returns = self.daily_results["return"].drop_nulls()
        equity = self.daily_results["portfolio_value"]
        
        total_return = float(equity[-1] / equity[0] - 1)
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        peak = equity.cum_max()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        
        # Sortino
        downside = returns.filter(returns < 0)
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 1
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0,
            "total_trades": len(self.trades),
            "total_costs": float(self.trades["cost"].sum()) if len(self.trades) > 0 else 0,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        s = self.summary()
        print(f"""
Backtest Results
================
Total Return:     {s['total_return']*100:>8.2f}%
Annual Return:    {s['annualized_return']*100:>8.2f}%
Annual Vol:       {s['annualized_volatility']*100:>8.2f}%
Sharpe Ratio:     {s['sharpe_ratio']:>8.2f}
Sortino Ratio:    {s['sortino_ratio']:>8.2f}
Max Drawdown:     {s['max_drawdown']*100:>8.2f}%
Calmar Ratio:     {s['calmar_ratio']:>8.2f}
Total Trades:     {s['total_trades']:>8,}
Total Costs:      ${s['total_costs']:>10,.2f}
""")


from harbinger.optimize import (
    build_factor_covariance,
    mean_variance_optimize,
    mean_variance_dynamic,
    calculate_active_risk,
)


def backtest(
    signals: pl.DataFrame,
    returns: pl.DataFrame,
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    benchmark_weights: pl.DataFrame,
    prices: pl.DataFrame,
    config: BacktestConfig = None,
) -> BacktestResult:
    """
    Run backtest with MVO and constraints.
    
    Pipeline:
    1. MVO optimization (unconstrained)
    2. Apply optimizer constraints
    3. Apply trading constraints
    4. Execute trades and compute PnL
    
    Args:
        signals: DataFrame[ticker, date, value]
        returns: DataFrame[ticker, date, return]
        factor_loadings: DataFrame[ticker, date, factor, loading]
        factor_covariances: DataFrame[date, factor_1, factor_2, covariance]
        idio_vol: DataFrame[ticker, date, idio_vol]
        benchmark_weights: DataFrame[ticker, date, weight]
        prices: DataFrame[ticker, date, close]
        config: BacktestConfig
    
    Returns:
        BacktestResult with daily PnL and returns
    """
    if config is None:
        config = BacktestConfig.default()
    
    # Get common dates
    signal_dates = set(signals["date"].unique().to_list())
    return_dates = set(returns["date"].unique().to_list())
    factor_dates = set(factor_loadings["date"].unique().to_list())
    idio_dates = set(idio_vol["date"].unique().to_list())
    bench_dates = set(benchmark_weights["date"].unique().to_list())
    price_dates = set(prices["date"].unique().to_list())
    
    common_dates = sorted(
        signal_dates & return_dates & factor_dates & idio_dates & bench_dates & price_dates
    )
    
    if len(common_dates) < 2:
        raise ValueError("Not enough common dates for backtest")
    
    # Forward returns: signal_T forms portfolio at close of T, captures return T→T+1
    returns_sorted = returns.sort(["ticker", "date"])
    forward_returns = returns_sorted.with_columns([
        pl.col("return").shift(-1).over("ticker").alias("forward_return")
    ]).select(["ticker", "date", "forward_return"]).drop_nulls()
    
    # Initialize
    portfolio_value = config.initial_capital
    current_weights: dict[str, float] = {}
    
    daily_results = []
    all_weights = []
    all_trades = []
    
    for i, date in enumerate(common_dates[:-1]):  # Exclude last date (no forward return)
        # Get data for this date
        day_signals = signals.filter(pl.col("date") == date)
        day_prices = prices.filter(pl.col("date") == date)
        day_bench = benchmark_weights.filter(pl.col("date") == date)
        day_forward = forward_returns.filter(pl.col("date") == date)
        
        if day_signals.is_empty() or day_prices.is_empty():
            continue
        
        # Build covariance matrix
        cov, cov_tickers = build_factor_covariance(
            factor_loadings, factor_covariances, idio_vol, date
        )
        
        if len(cov_tickers) == 0:
            continue
        
        # Get expected returns from signals
        expected_returns = {
            row["ticker"]: row["value"]
            for row in day_signals.iter_rows(named=True)
            if row["ticker"] in cov_tickers
        }
        
        if not expected_returns:
            continue
        
        # Get benchmark weights
        bench_dict = {
            row["ticker"]: row["weight"]
            for row in day_bench.iter_rows(named=True)
            if row["ticker"] in cov_tickers
        }
        
        # Get prices
        price_dict = {
            row["ticker"]: row["close"]
            for row in day_prices.iter_rows(named=True)
        }
        
        # Step 1: MVO optimization
        # Either fixed gamma or dynamic gamma with target active risk
        if config.gamma == "dynamic":
            # Dynamic gamma mode: find gamma that achieves target active risk
            if config.target_active_risk is None:
                raise ValueError("target_active_risk required when gamma='dynamic'")
            
            optimized_weights, _, _ = mean_variance_dynamic(
                expected_returns=expected_returns,
                cov_matrix=cov,
                tickers=cov_tickers,
                benchmark_weights=bench_dict,
                target_active_risk=config.target_active_risk,
            )
        else:
            # Fixed gamma mode
            optimized_weights = mean_variance_optimize(
                expected_returns=expected_returns,
                cov_matrix=cov,
                tickers=cov_tickers,
                gamma=config.gamma,
            )
        
        # Step 2: Apply optimizer constraints (e.g., MaxWeight)
        optimizer_context = {
            "cov_matrix": cov,
            "benchmark_weights": bench_dict,
            "tickers": cov_tickers,
            "prices": price_dict,
            "portfolio_value": portfolio_value,
        }
        
        optimized_weights = apply_optimizer_constraints(
            optimized_weights,
            config.optimizer_constraints,
            optimizer_context,
        )
        
        # Step 3: Apply trading constraints
        trading_context = {
            "portfolio_value": portfolio_value,
            "current_weights": current_weights,
            "prices": price_dict,
        }
        
        final_weights = apply_trading_constraints(
            optimized_weights,
            config.trading_constraints,
            trading_context,
        )
        
        # Calculate trades
        for ticker in set(final_weights.keys()) | set(current_weights.keys()):
            old_weight = current_weights.get(ticker, 0)
            new_weight = final_weights.get(ticker, 0)
            
            if abs(new_weight - old_weight) > 1e-10 and ticker in price_dict:
                trade_value = abs(new_weight - old_weight) * portfolio_value
                cost = trade_value * (config.slippage_bps + config.commission_bps) / 10000
                
                all_trades.append({
                    "date": date,
                    "ticker": ticker,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "value": trade_value,
                    "cost": cost,
                })
        
        # Record weights
        for ticker, weight in final_weights.items():
            all_weights.append({
                "date": date,
                "ticker": ticker,
                "weight": weight,
            })
        
        # Calculate portfolio return using forward returns
        forward_dict = {
            row["ticker"]: row["forward_return"]
            for row in day_forward.iter_rows(named=True)
        }
        
        portfolio_return = sum(
            final_weights.get(t, 0) * forward_dict.get(t, 0)
            for t in final_weights.keys()
        )
        
        # Deduct trading costs
        day_costs = sum(
            t["cost"] for t in all_trades if t["date"] == date
        )
        
        # Update portfolio value
        pnl = portfolio_value * portfolio_return - day_costs
        portfolio_value = portfolio_value + pnl
        
        # Calculate turnover
        turnover = sum(
            abs(final_weights.get(t, 0) - current_weights.get(t, 0))
            for t in set(final_weights.keys()) | set(current_weights.keys())
        ) / 2
        
        daily_results.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "pnl": pnl,
            "return": portfolio_return - day_costs / (portfolio_value - pnl) if portfolio_value != pnl else 0,
            "turnover": turnover,
        })
        
        current_weights = final_weights
    
    return BacktestResult(
        daily_results=pl.DataFrame(daily_results),
        weights=pl.DataFrame(all_weights) if all_weights else pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "weight": pl.Float64}),
        trades=pl.DataFrame(all_trades) if all_trades else pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "old_weight": pl.Float64, "new_weight": pl.Float64, "value": pl.Float64, "cost": pl.Float64}),
    )
