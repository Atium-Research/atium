"""Portfolio optimization primitives."""

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


def mean_variance_optimize(
    expected_returns: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> dict[str, float]:
    """Mean-variance optimization."""
    n = len(tickers)
    if n == 0:
        return {}

    mu = np.array([expected_returns.get(t, 0) for t in tickers])

    try:
        cov_inv = np.linalg.inv(cov_matrix)
        weights = (1.0 / risk_aversion) * cov_inv @ mu
    except np.linalg.LinAlgError:
        weights = np.ones(n) / n

    if long_only:
        weights = np.maximum(weights, 0)

    total = np.sum(np.abs(weights))
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(n) / n

    return dict(zip(tickers, weights))


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
    MVO with iterative scaling to target specific active risk.
    
    Iteratively adjusts scale factor to account for long-only constraint compression.
    
    Returns: (weights, scale_factor, achieved_active_risk)
    """
    n = len(tickers)
    if n == 0:
        return {}, 0.0, 0.0

    mu = np.array([expected_returns.get(t, 0) for t in tickers])
    bench = np.array([benchmark_weights.get(t, 0) for t in tickers])

    # Normalize benchmark weights
    if bench.sum() > 0:
        bench = bench / bench.sum()
    else:
        bench = np.ones(n) / n

    # Run MVO with low lambda to get aggressive direction
    lambda_ = 1.0
    try:
        cov_inv = np.linalg.inv(cov_matrix)
        raw_weights = (1.0 / lambda_) * cov_inv @ mu
    except np.linalg.LinAlgError:
        raw_weights = np.ones(n) / n

    # Normalize to get direction (sum of abs = 1)
    raw_sum = np.sum(np.abs(raw_weights))
    if raw_sum > 0:
        raw_weights = raw_weights / raw_sum

    # Active direction
    active_direction = raw_weights - bench

    # Iteratively find scale that achieves target after constraints
    scale = 1.0
    achieved_risk = 0.0

    for _ in range(max_iterations):
        # Apply scale
        scaled_active = active_direction * scale
        weights = bench + scaled_active

        # Apply long-only
        if long_only:
            weights = np.maximum(weights, 0)

        # Renormalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = bench.copy()

        # Calculate achieved risk
        achieved_risk = calculate_active_risk(weights, bench, cov_matrix)

        # Check convergence
        if abs(achieved_risk - target_active_risk) < tolerance:
            break

        # Adjust scale
        if achieved_risk > 0:
            # Scale up/down proportionally
            scale = scale * (target_active_risk / achieved_risk)
            scale = min(scale, 100.0)  # Cap scale to avoid explosion
        else:
            scale = scale * 2.0

    final_weights = dict(zip(tickers, weights))
    return final_weights, scale, achieved_risk


def _predict_lambda(data: list[tuple[float, float]], target_risk: float) -> float:
    """Predict lambda to achieve target risk using linear model."""
    data_arr = np.array(data)
    lambdas = data_arr[:, 0]
    sigmas = data_arr[:, 1]

    # Model: sigma = M / (2 * lambda) => M = 2 * lambda * sigma
    # Fit M via least squares: X = 1/(2*lambda), y = sigma
    X = 1.0 / (2.0 * lambdas)
    M = np.dot(X, sigmas) / np.dot(X, X)

    # Target: target_risk = M / (2 * lambda) => lambda = M / (2 * target_risk)
    return M / (2.0 * target_risk)


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
