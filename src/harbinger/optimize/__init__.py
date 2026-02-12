"""Portfolio optimization primitives."""

import numpy as np
import polars as pl


def equal_weight(tickers: list[str], date: str | None = None) -> dict[str, float]:
    """Generate equal weights for all tickers."""
    n = len(tickers)
    weight = 1.0 / n if n > 0 else 0.0
    return {ticker: weight for ticker in tickers}


def mean_variance(
    returns: pl.DataFrame,
    target_return: float | None = None,
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    """
    Mean-variance optimization.
    
    Args:
        returns: DataFrame with columns [ticker, date, return]
        target_return: Target portfolio return (optional)
        risk_aversion: Risk aversion parameter (higher = more conservative)
    
    Returns:
        Dictionary of ticker -> weight
    """
    # Pivot to wide format
    wide = returns.pivot(on="ticker", index="date", values="return").drop("date")
    tickers = wide.columns
    
    # Calculate expected returns and covariance
    returns_matrix = wide.to_numpy()
    
    # Drop any columns with NaN
    valid_mask = ~np.any(np.isnan(returns_matrix), axis=0)
    returns_matrix = returns_matrix[:, valid_mask]
    tickers = [t for t, v in zip(tickers, valid_mask) if v]
    
    if len(tickers) == 0:
        return {}
    
    mu = np.mean(returns_matrix, axis=0)
    cov = np.cov(returns_matrix.T)
    
    # Regularize covariance matrix
    cov = cov + np.eye(len(tickers)) * 1e-6
    
    # Simple mean-variance: w = (1/lambda) * Sigma^-1 * mu
    try:
        cov_inv = np.linalg.inv(cov)
        weights = (1.0 / risk_aversion) * cov_inv @ mu
    except np.linalg.LinAlgError:
        # Fallback to equal weight
        weights = np.ones(len(tickers)) / len(tickers)
    
    # Normalize to sum to 1 (long-only)
    weights = np.maximum(weights, 0)  # No shorts
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(len(tickers)) / len(tickers)
    
    return dict(zip(tickers, weights))


def generate_weights_series(
    returns: pl.DataFrame,
    method: str = "equal",
    lookback: int = 60,
    rebalance_freq: int = 1,
    **kwargs,
) -> pl.DataFrame:
    """
    Generate weight series over time.
    
    Args:
        returns: DataFrame with columns [ticker, date, return]
        method: "equal" or "mean_variance"
        lookback: Number of days to use for optimization
        rebalance_freq: Rebalance every N days
    
    Returns:
        DataFrame with columns [ticker, date, weight]
    """
    dates = sorted(returns["date"].unique().to_list())
    all_weights = []
    
    for i, date in enumerate(dates):
        # Skip if not rebalance day
        if i % rebalance_freq != 0:
            continue
        
        # Get lookback window
        lookback_dates = dates[max(0, i - lookback):i]
        if len(lookback_dates) < 20:  # Need minimum history
            continue
        
        window_returns = returns.filter(pl.col("date").is_in(lookback_dates))
        tickers = window_returns["ticker"].unique().to_list()
        
        if method == "equal":
            weights = equal_weight(tickers)
        elif method == "mean_variance":
            weights = mean_variance(window_returns, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for ticker, weight in weights.items():
            all_weights.append({
                "date": date,
                "ticker": ticker,
                "weight": weight,
            })
    
    return pl.DataFrame(all_weights)
