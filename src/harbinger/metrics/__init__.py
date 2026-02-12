"""Performance and risk metrics."""

import numpy as np
import polars as pl


def sharpe_ratio(returns: np.ndarray | pl.Series, risk_free: float = 0.0, annualize: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    if isinstance(returns, pl.Series):
        returns = returns.to_numpy()
    
    excess = returns - risk_free / annualize
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(annualize)


def sortino_ratio(returns: np.ndarray | pl.Series, risk_free: float = 0.0, annualize: int = 252) -> float:
    """Calculate annualized Sortino ratio (downside deviation)."""
    if isinstance(returns, pl.Series):
        returns = returns.to_numpy()
    
    excess = returns - risk_free / annualize
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.mean(excess) / np.std(downside) * np.sqrt(annualize)


def max_drawdown(equity_curve: np.ndarray | pl.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    if isinstance(equity_curve, pl.Series):
        equity_curve = equity_curve.to_numpy()
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.min(drawdown))


def total_return(equity_curve: np.ndarray | pl.Series) -> float:
    """Calculate total return from equity curve."""
    if isinstance(equity_curve, pl.Series):
        equity_curve = equity_curve.to_numpy()
    
    return float(equity_curve[-1] / equity_curve[0] - 1)


def annualized_return(returns: np.ndarray | pl.Series, periods: int = 252) -> float:
    """Calculate annualized return from daily returns."""
    if isinstance(returns, pl.Series):
        returns = returns.to_numpy()
    
    total = np.prod(1 + returns) - 1
    n_years = len(returns) / periods
    return (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0.0


def annualized_volatility(returns: np.ndarray | pl.Series, periods: int = 252) -> float:
    """Calculate annualized volatility from daily returns."""
    if isinstance(returns, pl.Series):
        returns = returns.to_numpy()
    
    return float(np.std(returns) * np.sqrt(periods))


def calmar_ratio(returns: np.ndarray | pl.Series, equity_curve: np.ndarray | pl.Series) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    ann_ret = annualized_return(returns)
    mdd = abs(max_drawdown(equity_curve))
    return ann_ret / mdd if mdd != 0 else 0.0


def calculate_metrics(returns: np.ndarray | pl.Series, equity_curve: np.ndarray | pl.Series) -> dict:
    """Calculate all standard metrics."""
    return {
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve),
    }
