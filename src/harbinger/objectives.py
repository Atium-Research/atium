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
        return cp.Maximize(weights @ alphas - self.lambda_ * 0.5 * weights.T @ covariance_matrix @ weights)


class MaxUtilityWithTargetActiveRisk(Objective):
    def __init__(self, target_active_risk: float):
        self.target_active_risk = target_active_risk

    def build(
        self, weights: cp.Variable, **kwargs
    ) -> cp.Expression:
        pass
