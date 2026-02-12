"""Portfolio optimization primitives."""

import numpy as np
import polars as pl


def build_factor_covariance(
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    date = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build covariance matrix using factor model: Σ = B * F * B' + D
    
    Args:
        factor_loadings: DataFrame with [ticker, (date,) factor, loading]
        factor_covariances: DataFrame with [(date,) factor_1, factor_2, covariance]
        idio_vol: DataFrame with [ticker, (date,) idio_vol]
        date: Optional date to filter to (if data has date column)
    
    Returns:
        - covariance matrix (n_assets x n_assets)
        - list of tickers in order
    """
    # Filter to specific date if provided and date column exists
    if date is not None:
        if "date" in factor_loadings.columns:
            factor_loadings = factor_loadings.filter(pl.col("date") == date)
        if "date" in factor_covariances.columns:
            factor_covariances = factor_covariances.filter(pl.col("date") == date)
        if "date" in idio_vol.columns:
            idio_vol = idio_vol.filter(pl.col("date") == date)
    
    # Get tickers that have both loadings and idio vol
    tickers_with_loadings = set(factor_loadings["ticker"].unique().to_list())
    tickers_with_idio = set(idio_vol["ticker"].unique().to_list())
    tickers = sorted(tickers_with_loadings & tickers_with_idio)
    
    if not tickers:
        return np.array([[]]), []
    
    # Filter to common tickers
    factor_loadings = factor_loadings.filter(pl.col("ticker").is_in(tickers))
    idio_vol = idio_vol.filter(pl.col("ticker").is_in(tickers))
    
    # Get factors
    factors = sorted(factor_loadings["factor"].unique().to_list())
    n_assets = len(tickers)
    n_factors = len(factors)
    
    # Build B matrix (n_assets x n_factors)
    B = np.zeros((n_assets, n_factors))
    loadings_pivot = factor_loadings.pivot(on="factor", index="ticker", values="loading").sort("ticker")
    for i, factor in enumerate(factors):
        if factor in loadings_pivot.columns:
            B[:, i] = loadings_pivot[factor].fill_null(0).to_numpy()
    
    # Build F matrix (n_factors x n_factors)
    F = np.zeros((n_factors, n_factors))
    for row in factor_covariances.iter_rows(named=True):
        f1, f2, cov = row["factor_1"], row["factor_2"], row["covariance"]
        if f1 in factors and f2 in factors:
            i, j = factors.index(f1), factors.index(f2)
            F[i, j] = cov
            F[j, i] = cov  # Symmetric
    
    # Build D matrix (diagonal idiosyncratic variance)
    idio_sorted = idio_vol.sort("ticker")
    idio_var = idio_sorted["idio_vol"].fill_null(0.02).to_numpy() ** 2  # vol -> variance
    D = np.diag(idio_var)
    
    # Σ = B * F * B' + D
    cov_matrix = B @ F @ B.T + D
    
    # Regularize
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6
    
    return cov_matrix, tickers


def mean_variance_optimize(
    expected_returns: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> dict[str, float]:
    """
    Mean-variance optimization: max w'μ - (λ/2) w'Σw
    
    For long-only, we use a simple iterative approach.
    """
    n = len(tickers)
    if n == 0:
        return {}
    
    # Build expected returns vector
    mu = np.array([expected_returns.get(t, 0) for t in tickers])
    
    # Unconstrained solution: w = (1/λ) * Σ^-1 * μ
    try:
        cov_inv = np.linalg.inv(cov_matrix)
        weights = (1.0 / risk_aversion) * cov_inv @ mu
    except np.linalg.LinAlgError:
        weights = np.ones(n) / n
    
    if long_only:
        # Simple approach: zero out negative weights, renormalize
        weights = np.maximum(weights, 0)
    
    # Normalize to sum to 1
    total = np.sum(np.abs(weights))
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(n) / n
    
    return dict(zip(tickers, weights))


def signal_to_expected_returns(
    signals: pl.DataFrame,
    date: str,
    scale: float = 1.0,
) -> dict[str, float]:
    """
    Convert signals to expected returns.
    
    Args:
        signals: DataFrame with [ticker, date, value]
        date: Date to get signals for
        scale: Scaling factor for signals
    """
    if isinstance(date, str):
        from datetime import datetime
        date = datetime.strptime(date, "%Y-%m-%d").date()
    
    day_signals = signals.filter(pl.col("date") == date)
    return {
        row["ticker"]: row["value"] * scale
        for row in day_signals.iter_rows(named=True)
    }
