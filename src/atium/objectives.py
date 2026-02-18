from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np


class Objective(ABC):
    """Base class for portfolio optimization objectives."""

    @abstractmethod
    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        """Build and return a CVXPY objective expression.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional data (alphas, covariance_matrix, benchmark_weights).
        """
        pass


class MaxUtility(Objective):
    """Mean-variance utility: maximize w @ alpha - (lambda/2) * w' Sigma w.

    Args:
        lambda_: Risk-aversion parameter. Higher values penalize variance more.
    """

    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        """Return the mean-variance objective expression."""
        alphas: np.ndarray = kwargs.get('alphas')
        covariance_matrix: np.ndarray = kwargs.get('covariance_matrix')
        return cp.Maximize(weights @ alphas - self.lambda_ * 0.5 * cp.quad_form(weights, covariance_matrix))


class MaxUtilityWithTargetActiveRisk(Objective):
    """Mean-variance utility that iteratively calibrates lambda to hit a target active risk.

    Solves the optimization multiple times, using linear regression on
    (1/2*lambda, active_risk) pairs to predict the lambda that produces
    the desired annualized tracking error vs. the benchmark.

    Args:
        target_active_risk: Desired annualized active risk (e.g. 0.05 for 5%).
    """

    def __init__(self, target_active_risk: float):
        self.target_active_risk = target_active_risk

    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        """Calibrate lambda to match the target active risk, then return the objective."""
        alphas: np.ndarray = kwargs.get('alphas')
        covariance_matrix: np.ndarray = kwargs.get('covariance_matrix')
        benchmark_weights: np.ndarray = kwargs.get('benchmark_weights')
        optimizer_constraints = kwargs.get('constraints', [])

        n_assets = len(alphas)
        lambda_ = None
        active_risk = float('inf')
        error = 0.005
        max_iterations = 5
        iterations = 1
        data = []

        while abs(active_risk - self.target_active_risk) > error:
            if lambda_ is None:
                lambda_ = 100
            else:
                lambda_ = self._predict_lambda(data, self.target_active_risk)

            w = cp.Variable(n_assets)
            obj = cp.Maximize(w @ alphas - 0.5 * lambda_ * cp.quad_form(w, covariance_matrix))
            constraints = [c.build(w) for c in optimizer_constraints]
            prob = cp.Problem(obj, constraints)
            prob.solve()

            active_w = w.value - benchmark_weights
            active_risk = float(np.sqrt(active_w @ covariance_matrix @ active_w) * np.sqrt(252))

            data.append((lambda_, active_risk))

            if iterations >= max_iterations:
                break
            else:
                iterations += 1

        return cp.Maximize(weights @ alphas - 0.5 * lambda_ * cp.quad_form(weights, covariance_matrix))

    @staticmethod
    def _predict_lambda(data: list[tuple[float, float]], active_risk: float) -> float:
        """Predict the lambda that will produce the target active risk.

        Uses a linear model fit on 1/(2*lambda) vs. observed active risk
        from prior iterations.
        """
        data_np = np.array(data)
        lambda_ = data_np[:, 0]
        sigma = data_np[:, 1]
        X = 1 / (2 * lambda_)
        M = np.dot(X, sigma) / np.dot(X, X)
        return M / (2 * active_risk)
