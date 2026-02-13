"""
Harbinger Backtester

Single entry point for running backtests with MVO and trading constraints.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    
    initial_capital: float = 100_000.0
    target_active_risk: float = 0.05
    risk_aversion: float = 1.0
    
    # Trading constraints
    min_position_value: float = 1.0
    max_position_weight: float = 0.10
    max_turnover: float = 1.0
    long_only: bool = True
    
    # Costs
    slippage_bps: float = 5.0
    commission_bps: float = 10.0


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


def build_covariance_matrix(
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    date,
) -> tuple[np.ndarray, list[str]]:
    """Build factor model covariance matrix for a specific date."""
    
    # Filter to date
    fl = factor_loadings.filter(pl.col("date") == date)
    fc = factor_covariances.filter(pl.col("date") == date)
    iv = idio_vol.filter(pl.col("date") == date)
    
    # Get common tickers
    tickers_fl = set(fl["ticker"].unique().to_list())
    tickers_iv = set(iv["ticker"].unique().to_list())
    tickers = sorted(tickers_fl & tickers_iv)
    
    if not tickers:
        return np.array([[]]), []
    
    fl = fl.filter(pl.col("ticker").is_in(tickers))
    iv = iv.filter(pl.col("ticker").is_in(tickers))
    
    factors = sorted(fl["factor"].unique().to_list())
    n_assets = len(tickers)
    n_factors = len(factors)
    
    # Build B matrix (loadings)
    B = np.zeros((n_assets, n_factors))
    loadings_pivot = fl.pivot(on="factor", index="ticker", values="loading").sort("ticker")
    for i, factor in enumerate(factors):
        if factor in loadings_pivot.columns:
            B[:, i] = loadings_pivot[factor].fill_null(0).to_numpy()
    
    # Build F matrix (factor covariance)
    F = np.zeros((n_factors, n_factors))
    for row in fc.iter_rows(named=True):
        f1, f2, cov = row["factor_1"], row["factor_2"], row["covariance"]
        if f1 in factors and f2 in factors:
            i, j = factors.index(f1), factors.index(f2)
            F[i, j] = cov
            F[j, i] = cov
    
    # Build D matrix (idio variance)
    iv_sorted = iv.sort("ticker")
    idio_var = iv_sorted["idio_vol"].fill_null(0.02).to_numpy() ** 2
    D = np.diag(idio_var)
    
    # Σ = B * F * B' + D
    cov = B @ F @ B.T + D + np.eye(n_assets) * 1e-6
    
    return cov, tickers


def calculate_active_risk(
    weights: np.ndarray,
    benchmark: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Calculate annualized active risk."""
    active = weights - benchmark
    var = active @ cov @ active
    return float(np.sqrt(var) * np.sqrt(252))


def optimize_portfolio(
    expected_returns: dict[str, float],
    cov: np.ndarray,
    tickers: list[str],
    benchmark: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    """
    Run MVO with risk targeting.
    
    1. Compute raw MVO weights
    2. Scale to target active risk
    3. Apply long-only constraint
    """
    n = len(tickers)
    if n == 0:
        return {}
    
    mu = np.array([expected_returns.get(t, 0) for t in tickers])
    bench = np.array([benchmark.get(t, 0) for t in tickers])
    
    # Normalize benchmark
    if bench.sum() > 0:
        bench = bench / bench.sum()
    else:
        bench = np.ones(n) / n
    
    # MVO: w* = (1/λ) * Σ^-1 * μ
    try:
        cov_inv = np.linalg.inv(cov)
        raw = (1.0 / config.risk_aversion) * cov_inv @ mu
    except np.linalg.LinAlgError:
        raw = np.ones(n) / n
    
    # Normalize raw weights
    raw_sum = np.sum(np.abs(raw))
    if raw_sum > 0:
        raw = raw / raw_sum
    
    # Scale to target active risk (iterative due to long-only constraint)
    active_dir = raw - bench
    scale = 1.0
    
    for _ in range(15):
        scaled = active_dir * scale
        weights = bench + scaled
        
        if config.long_only:
            weights = np.maximum(weights, 0)
        
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        
        achieved_risk = calculate_active_risk(weights, bench, cov)
        
        if abs(achieved_risk - config.target_active_risk) < 0.002:
            break
        
        if achieved_risk > 0:
            scale = scale * (config.target_active_risk / achieved_risk)
            scale = min(scale, 100.0)
    
    return dict(zip(tickers, weights))


def apply_trading_constraints(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    prices: dict[str, float],
    portfolio_value: float,
    config: BacktestConfig,
) -> dict[str, float]:
    """
    Apply trading constraints to target weights.
    
    1. Cap position weights
    2. Filter out positions below min value
    3. Cap turnover
    """
    constrained = {}
    
    for ticker, weight in target_weights.items():
        # Skip if no price
        if ticker not in prices:
            continue
        
        # Cap position weight
        weight = min(weight, config.max_position_weight)
        if config.long_only:
            weight = max(weight, 0)
        else:
            weight = max(weight, -config.max_position_weight)
        
        # Check min position value
        position_value = abs(weight * portfolio_value)
        if position_value < config.min_position_value:
            continue
        
        constrained[ticker] = weight
    
    # Renormalize
    total = sum(constrained.values())
    if total > 0:
        constrained = {k: v / total for k, v in constrained.items()}
    
    # Calculate turnover and cap if needed
    all_tickers = set(constrained.keys()) | set(current_weights.keys())
    turnover = sum(
        abs(constrained.get(t, 0) - current_weights.get(t, 0))
        for t in all_tickers
    ) / 2  # Divide by 2 for one-way turnover
    
    if turnover > config.max_turnover and turnover > 0:
        # Blend toward target
        blend = config.max_turnover / turnover
        constrained = {
            t: current_weights.get(t, 0) + blend * (constrained.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        }
        # Clean up zeros
        constrained = {k: v for k, v in constrained.items() if abs(v) > 1e-10}
    
    return constrained


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
    Run backtest with MVO and trading constraints.
    
    Data alignment:
    - Signal from day T is used to form portfolio at close of day T
    - Forward return (T to T+1) is used to compute PnL
    
    Args:
        signals: DataFrame[ticker, date, value]
        returns: DataFrame[ticker, date, return] - daily returns
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
        config = BacktestConfig()
    
    # LAG SIGNALS INTERNALLY
    # Signal from day T is used to trade on day T+1
    # This avoids look-ahead bias since signal_T uses data from day T
    signals_lagged = signals.sort(["ticker", "date"]).with_columns([
        pl.col("date").shift(-1).over("ticker").alias("trade_date")
    ]).drop("date").rename({"trade_date": "date"}).drop_nulls()
    
    # Get common dates across all data (using lagged signal dates)
    signal_dates = set(signals_lagged["date"].unique().to_list())
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
    
    # Forward returns: we need return from T to T+1 for portfolio held at close of T
    # Since signals are lagged (signal computed at T is used at T+1),
    # we need forward return from T+1 = return realized from T+1 to T+2
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
    
    for i, date in enumerate(common_dates[:-1]):  # Exclude last date
        # Get data for this date (using lagged signals)
        day_signals = signals_lagged.filter(pl.col("date") == date)
        day_prices = prices.filter(pl.col("date") == date)
        day_bench = benchmark_weights.filter(pl.col("date") == date)
        day_forward = forward_returns.filter(pl.col("date") == date)
        
        if day_signals.is_empty() or day_prices.is_empty():
            continue
        
        # Build covariance matrix
        cov, cov_tickers = build_covariance_matrix(
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
        target_weights = optimize_portfolio(
            expected_returns, cov, cov_tickers, bench_dict, config
        )
        
        # Step 2: Apply trading constraints
        final_weights = apply_trading_constraints(
            target_weights, current_weights, price_dict, portfolio_value, config
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
        # Signal_T (lagged to T+1) forms portfolio at close of T+1
        # Captures return from T+1 to T+2 = forward_return at T+1
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
