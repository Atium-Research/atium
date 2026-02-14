"""
Optimizer constraints.

These are convex constraints that go into the QP solver.
They are enforced during optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class OptimizerConstraint(ABC):
    """Base class for optimizer constraints."""
    
    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """
        Return list of DataAdapter methods this constraint requires.
        """
        pass
    
    @abstractmethod
    def apply(self, problem: "OptimizationProblem", context: dict[str, Any]) -> None:
        """
        Add this constraint to the optimization problem.
        
        Args:
            problem: The optimization problem being built
            context: Dict containing data for this date
        """
        pass


# =============================================================================
# Basic Constraints
# =============================================================================

@dataclass
class LongOnly(OptimizerConstraint):
    """No short positions allowed (w >= 0)."""
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("long_only", lower_bound=0.0)


@dataclass
class FullyInvested(OptimizerConstraint):
    """Weights must sum to 1."""
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("sum_to_one", sum_equals=1.0)


@dataclass
class MaxWeight(OptimizerConstraint):
    """Maximum weight per position."""
    
    max_weight: float = 0.10
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("max_weight", upper_bound=self.max_weight)


@dataclass
class MinWeight(OptimizerConstraint):
    """Minimum weight per position (for shorts)."""
    
    min_weight: float = -0.10
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("min_weight", lower_bound=self.min_weight)


# =============================================================================
# Exposure Constraints
# =============================================================================

@dataclass
class MaxGrossExposure(OptimizerConstraint):
    """Maximum gross exposure (sum of absolute weights)."""
    
    max_exposure: float = 2.0  # 200% gross
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("max_gross_exposure", abs_sum_max=self.max_exposure)


@dataclass
class DollarNeutral(OptimizerConstraint):
    """Long dollar value equals short dollar value (sum of weights = 0)."""
    
    tolerance: float = 0.01  # Allow 1% deviation
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        problem.add_constraint("dollar_neutral", sum_equals=0.0, tolerance=self.tolerance)


@dataclass 
class BetaNeutral(OptimizerConstraint):
    """Portfolio beta equals zero."""
    
    tolerance: float = 0.01
    
    def get_dependencies(self) -> list[str]:
        return ["get_factor_exposures"]  # Need market beta
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        betas = context.get("factor_exposures", {})
        market_betas = {t: exp.get("market", 1.0) for t, exp in betas.items()}
        problem.add_constraint(
            "beta_neutral",
            weighted_sum_equals=0.0,
            weights=market_betas,
            tolerance=self.tolerance,
        )


# =============================================================================
# Sector Constraints
# =============================================================================

@dataclass
class MaxSectorWeight(OptimizerConstraint):
    """Maximum weight in any single sector."""
    
    max_weight: float = 0.25
    
    def get_dependencies(self) -> list[str]:
        return ["get_sector_mapping"]
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        sector_mapping = context["sector_mapping"]
        problem.add_constraint(
            "max_sector_weight",
            group_max=self.max_weight,
            groups=sector_mapping,
        )


@dataclass
class SectorNeutral(OptimizerConstraint):
    """Net zero exposure within each sector."""
    
    tolerance: float = 0.01
    
    def get_dependencies(self) -> list[str]:
        return ["get_sector_mapping"]
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        sector_mapping = context["sector_mapping"]
        problem.add_constraint(
            "sector_neutral",
            group_sum_equals=0.0,
            groups=sector_mapping,
            tolerance=self.tolerance,
        )


# =============================================================================
# Factor Constraints
# =============================================================================

@dataclass
class MaxFactorExposure(OptimizerConstraint):
    """Maximum exposure to a specific factor."""
    
    factor: str
    max_exposure: float
    
    def get_dependencies(self) -> list[str]:
        return ["get_factor_exposures"]
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        factor_exposures = context["factor_exposures"]
        exposures = {t: exp.get(self.factor, 0.0) for t, exp in factor_exposures.items()}
        problem.add_constraint(
            f"max_{self.factor}_exposure",
            weighted_sum_max=self.max_exposure,
            weights=exposures,
        )


@dataclass
class FactorNeutral(OptimizerConstraint):
    """Neutral exposure to a specific factor."""
    
    factor: str
    tolerance: float = 0.01
    
    def get_dependencies(self) -> list[str]:
        return ["get_factor_exposures"]
    
    def apply(self, problem, context: dict[str, Any]) -> None:
        factor_exposures = context["factor_exposures"]
        exposures = {t: exp.get(self.factor, 0.0) for t, exp in factor_exposures.items()}
        problem.add_constraint(
            f"{self.factor}_neutral",
            weighted_sum_equals=0.0,
            weights=exposures,
            tolerance=self.tolerance,
        )
