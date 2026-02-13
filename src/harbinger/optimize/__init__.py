"""Portfolio optimization primitives."""

from typing import Literal

import cvxpy as cp
import numpy as np
import polars as pl


def build_factor_covariance(
    factor_loadings: pl.DataFrame,
    factor_covariances: pl.DataFrame,
    idio_vol: pl.DataFrame,
    date=None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build covariance matrix using factor model: Σ = B * F * B' + D
    """
    # Filter to specific date if provided
    if date is not None:
        if "date" in factor_loadings.columns:
            factor_loadings = factor_loadings.filter(pl.col("date") == date)
        if "date" in factor_covariances.columns:
            factor_covariances = factor_covariances.filter(pl.col("date") == date)
        if "date" in idio_vol.columns:
            idio_vol = idio_vol.filter(pl.col("date") == date)

    # Get common tickers
    tickers_with_loadings = set(factor_loadings["ticker"].unique().to_list())
    tickers_with_idio = set(idio_vol["ticker"].unique().to_list())
    tickers = sorted(tickers_with_loadings & tickers_with_idio)

    if not tickers:
        return np.array([[]]), []

    factor_loadings = factor_loadings.filter(pl.col("ticker").is_in(tickers))
    idio_vol = idio_vol.filter(pl.col("ticker").is_in(tickers))

    factors = sorted(factor_loadings["factor"].unique().to_list())
    n_assets = len(tickers)
    n_factors = len(factors)

    # Build B matrix
    B = np.zeros((n_assets, n_factors))
    loadings_pivot = factor_loadings.pivot(on="factor", index="ticker", values="loading").sort("ticker")
    for i, factor in enumerate(factors):
        if factor in loadings_pivot.columns:
            B[:, i] = loadings_pivot[factor].fill_null(0).to_numpy()

    # Build F matrix
    F = np.zeros((n_factors, n_factors))
    for row in factor_covariances.iter_rows(named=True):
        f1, f2, cov = row["factor_1"], row["factor_2"], row["covariance"]
        if f1 in factors and f2 in factors:
            i, j = factors.index(f1), factors.index(f2)
            F[i, j] = cov
            F[j, i] = cov

    # Build D matrix (idio variance)
    idio_sorted = idio_vol.sort("ticker")
    idio_var = idio_sorted["idio_vol"].fill_null(0.02).to_numpy() ** 2
    D = np.diag(idio_var)

    # Σ = B * F * B' + D
    cov_matrix = B @ F @ B.T + D
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6

    return cov_matrix, tickers


def calculate_active_risk(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    cov_matrix: np.ndarray,
    annualize: bool = True,
) -> float:
    """
    Calculate annualized active risk (tracking error).
    
    active_risk = sqrt(active_weights' * Σ * active_weights) * sqrt(252)
    """
    active_weights = weights - benchmark_weights
    variance = active_weights @ cov_matrix @ active_weights
    vol = np.sqrt(variance)
    if annualize:
        vol *= np.sqrt(252)
    return float(vol)


def solve_mvo_cvxpy(
    alphas: np.ndarray,
    cov_matrix: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Solve MVO using cvxpy with constraints.
    
    Max: alpha' @ w - (gamma / 2) * w' @ Σ @ w
    s.t. sum(w) = 1, w >= 0
    """
    n = len(alphas)
    weights = cp.Variable(n)

    objective = cp.Maximize(
        cp.matmul(weights, alphas)
        - 0.5 * gamma * cp.quad_form(weights, cov_matrix)
    )

    constraints = [
        cp.sum(weights) == 1,  # Full investment
        weights >= 0,  # Long only
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


def mean_variance_optimize(
    expected_returns: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    gamma: float = 1.0,
) -> dict[str, float]:
    """
    Mean-variance optimization using cvxpy.
    
    Args:
        expected_returns: Dict mapping ticker -> expected return (alpha)
        cov_matrix: Covariance matrix (n x n)
        tickers: List of tickers (defines ordering)
        gamma: Risk aversion parameter
    
    Returns:
        Dict mapping ticker -> optimal weight
    """
    n = len(tickers)
    if n == 0:
        return {}

    alphas = np.array([expected_returns.get(t, 0) for t in tickers])
    
    weights = solve_mvo_cvxpy(alphas, cov_matrix, gamma)
    
    if weights is None:
        # Fallback to equal weight if solver fails
        weights = np.ones(n) / n

    return dict(zip(tickers, weights))


def _predict_gamma(data: list[tuple[float, float]], target_risk: float) -> float:
    """
    Predict gamma to achieve target active risk using linear model.
    
    Based on the relationship: active_risk ≈ M / (2 * gamma)
    where M is fitted from observed (gamma, active_risk) pairs.
    
    Args:
        data: List of (gamma, active_risk) tuples from previous iterations
        target_risk: Target active risk to achieve
    
    Returns:
        Predicted gamma value
    """
    data_arr = np.array(data)
    gammas = data_arr[:, 0]
    sigmas = data_arr[:, 1]

    # Model: sigma = M / (2 * gamma) => M = 2 * gamma * sigma
    # Fit M via least squares: X = 1/(2*gamma), y = sigma
    X = 1.0 / (2.0 * gammas)
    M = np.dot(X, sigmas) / np.dot(X, X)

    # Target: target_risk = M / (2 * gamma) => gamma = M / (2 * target_risk)
    return M / (2.0 * target_risk)


def mean_variance_dynamic(
    expected_returns: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    benchmark_weights: dict[str, float],
    target_active_risk: float = 0.05,
    max_iterations: int = 5,
    tolerance: float = 0.005,
) -> tuple[dict[str, float], float, float]:
    """
    MVO with dynamic gamma to target specific active risk.
    
    Iteratively adjusts gamma (risk aversion) to achieve target active risk.
    Uses cvxpy solver at each iteration.
    
    This is the approach from nt-backtester: rather than scaling weights post-hoc,
    we find the gamma that produces the desired active risk through the optimizer.
    
    Args:
        expected_returns: Dict mapping ticker -> expected return (alpha)
        cov_matrix: Covariance matrix (n x n)
        tickers: List of tickers (defines ordering)
        benchmark_weights: Dict mapping ticker -> benchmark weight
        target_active_risk: Target annualized active risk (tracking error)
        max_iterations: Maximum iterations to find optimal gamma
        tolerance: Acceptable error in active risk
    
    Returns:
        (weights, final_gamma, achieved_active_risk)
    """
    n = len(tickers)
    if n == 0:
        return {}, 0.0, 0.0

    alphas = np.array([expected_returns.get(t, 0) for t in tickers])
    bench = np.array([benchmark_weights.get(t, 0) for t in tickers])
    
    # Normalize benchmark weights
    if bench.sum() > 0:
        bench = bench / bench.sum()
    else:
        bench = np.ones(n) / n

    active_risk = float("inf")
    gamma = None
    data: list[tuple[float, float]] = []
    
    for iteration in range(max_iterations):
        # First iteration: start with gamma=100, then predict
        if gamma is None:
            gamma = 100.0
        else:
            gamma = _predict_gamma(data, target_active_risk)
            # Clamp gamma to reasonable range
            gamma = max(1.0, min(gamma, 10000.0))

        # Solve MVO with current gamma
        raw_weights = solve_mvo_cvxpy(alphas, cov_matrix, gamma)
        
        if raw_weights is None:
            # Solver failed, fall back to equal weight
            raw_weights = np.ones(n) / n

        # Calculate active risk
        active_risk = calculate_active_risk(raw_weights, bench, cov_matrix)
        
        # Record data point for prediction
        data.append((gamma, active_risk))
        
        # Check convergence
        if abs(active_risk - target_active_risk) <= tolerance:
            break

    final_weights = dict(zip(tickers, raw_weights))
    return final_weights, gamma, active_risk


def mean_variance_with_risk_target(
    expected_returns: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    benchmark_weights: dict[str, float],
    target_active_risk: float = 0.05,
    long_only: bool = True,
    max_iterations: int = 15,
    tolerance: float = 0.002,
) -> tuple[dict[str, float], float, float]:
    """
    DEPRECATED: Use mean_variance_dynamic instead.
    
    MVO with iterative scaling to target specific active risk.
    This version scales weights post-hoc rather than finding optimal gamma.
    
    Returns: (weights, scale_factor, achieved_active_risk)
    """
    # Redirect to dynamic version
    return mean_variance_dynamic(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        tickers=tickers,
        benchmark_weights=benchmark_weights,
        target_active_risk=target_active_risk,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def signal_to_expected_returns(
    signals: pl.DataFrame,
    date,
    scale: float = 1.0,
) -> dict[str, float]:
    """Convert signals to expected returns."""
    day_signals = signals.filter(pl.col("date") == date)
    return {
        row["ticker"]: row["value"] * scale
        for row in day_signals.iter_rows(named=True)
    }
