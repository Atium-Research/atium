from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np


class Objective(ABC):
    @abstractmethod
    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        pass


class MaxUtility(Objective):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        alphas: np.ndarray = kwargs.get('alphas')
        covariance_matrix: np.ndarray = kwargs.get('covariance_matrix')
        return cp.Maximize(weights @ alphas - self.lambda_ * 0.5 * cp.quad_form(weights, covariance_matrix))


class MaxUtilityWithTargetActiveRisk(Objective):
    def __init__(self, target_active_risk: float):
        self.target_active_risk = target_active_risk

    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        alphas: np.ndarray = kwargs.get('alphas')
        covariance_matrix: np.ndarray = kwargs.get('covariance_matrix')
        benchmark_weights: np.ndarray = kwargs.get('benchmark_weights')

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
            constraints = [cp.sum(w) == 1, w >= 0]
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
        data_np = np.array(data)
        lambda_ = data_np[:, 0]
        sigma = data_np[:, 1]
        X = 1 / (2 * lambda_)
        M = np.dot(X, sigma) / np.dot(X, X)
        return M / (2 * active_risk)
