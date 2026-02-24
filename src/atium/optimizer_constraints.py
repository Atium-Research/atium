from abc import ABC, abstractmethod

import cvxpy as cp


class OptimizerConstraint(ABC):
    """Base class for hard constraints passed to the CVXPY solver."""

    @abstractmethod
    def build(self, weights: cp.Variable, **kwargs):
        """Return a CVXPY constraint expression."""
        pass


class LongOnly(OptimizerConstraint):
    """Constrain all portfolio weights to be non-negative (no short selling)."""

    def build(self, weights: cp.Variable, **kwargs):
        """Return the constraint w >= 0."""
        return weights >= 0


class FullyInvested(OptimizerConstraint):
    """Constrain portfolio weights to sum to one (fully deployed capital)."""

    def build(self, weights: cp.Variable, **kwargs):
        """Return the constraint sum(w) == 1."""
        return cp.sum(weights) == 1


class TargetBeta(OptimizerConstraint):
    """Constrain the portfolio beta to equal a target value.

    Enforces w @ betas == target_beta, where betas is a vector of
    per-asset betas passed via kwargs at solve time.

    Args:
        target_beta: Desired portfolio beta (e.g. 1.0 for market beta).
    """

    def __init__(self, target_beta: float):
        self.target_beta = target_beta

    def build(self, weights: cp.Variable, **kwargs):
        """Return the constraint w @ betas == target_beta."""
        betas = kwargs.get('betas')
        if betas is None:
            raise ValueError(
                "TargetBeta constraint requires 'betas' in kwargs. "
                "Ensure a BetaProvider is configured in the strategy."
            )
        return weights @ betas == self.target_beta
