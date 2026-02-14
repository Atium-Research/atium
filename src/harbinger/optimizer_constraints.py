from abc import ABC, abstractmethod
import cvxpy as cp


class OptimizerConstraint(ABC):
    @abstractmethod
    def build(self, weights: cp.Variable, **kwargs):
        pass


class LongOnly(OptimizerConstraint):
    def build(self, weights: cp.Variable, **kwargs):
        return weights >= 0


class FullyInvested(OptimizerConstraint):
    def build(self, weights: cp.Variable, **kwargs):
        return cp.sum(weights) == 1
