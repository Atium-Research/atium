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
