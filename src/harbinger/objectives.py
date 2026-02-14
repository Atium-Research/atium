"""
Objective functions for portfolio optimization.

Objectives define what the optimizer is trying to achieve.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class ObjectiveBase(ABC):
    """Base class for all objectives."""
    
    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """
        Return list of DataAdapter methods this objective requires.
        
        Returns:
            List of method names (e.g., ["get_benchmark_weights"])
        """
        pass
    
    @abstractmethod
    def build(self, context: dict[str, Any]) -> "ObjectiveExpr":
        """
        Build the objective expression for the optimizer.
        
        Args:
            context: Dict containing alphas, covariance, benchmark_weights, etc.
            
        Returns:
            ObjectiveExpr that can be optimized
        """
        pass


@dataclass
class ObjectiveExpr:
    """
    Expression representing the optimization problem.
    
    The optimizer will maximize: linear_term - 0.5 * gamma * quadratic_term
    subject to constraints.
    """
    linear_coef: np.ndarray      # Coefficients for linear term (alphas)
    quadratic_matrix: np.ndarray  # Matrix for quadratic term (covariance)
    gamma: float                  # Risk aversion parameter
    
    # For dynamic gamma objectives
    dynamic_gamma: bool = False
    target_active_risk: float | None = None
    benchmark_weights: np.ndarray | None = None


# =============================================================================
# Simple Objectives
# =============================================================================

@dataclass
class MaximizeAlpha(ObjectiveBase):
    """
    Maximize expected returns (alpha).
    
    Typically combined with a risk term or risk constraint.
    """
    weight: float = 1.0
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        alphas = context["alphas"]
        cov = context["covariance"]
        
        return ObjectiveExpr(
            linear_coef=alphas * self.weight,
            quadratic_matrix=cov,
            gamma=0.0,  # No risk penalty
        )


@dataclass
class MinimizeVariance(ObjectiveBase):
    """
    Minimize portfolio variance.
    
    Can be used alone (minimum variance portfolio) or with a weight
    as part of a composite objective.
    """
    weight: float = 1.0
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        n = len(context["tickers"])
        cov = context["covariance"]
        
        return ObjectiveExpr(
            linear_coef=np.zeros(n),
            quadratic_matrix=cov,
            gamma=self.weight,
        )


@dataclass
class MaximizeSharpe(ObjectiveBase):
    """
    Maximize Sharpe ratio (tangency portfolio).
    
    This finds the portfolio with highest risk-adjusted return.
    No benchmark is required.
    """
    risk_free_rate: float = 0.0
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        alphas = context["alphas"]
        cov = context["covariance"]
        
        # Adjust alphas for risk-free rate
        excess_returns = alphas - self.risk_free_rate
        
        # MaxSharpe is solved by finding the tangency portfolio
        # This is equivalent to maximizing alpha/vol, but we solve it
        # as a QP by iterating on gamma
        return ObjectiveExpr(
            linear_coef=excess_returns,
            quadratic_matrix=cov,
            gamma=1.0,  # Will be adjusted iteratively
            dynamic_gamma=True,  # Signals we need to find optimal gamma
        )


@dataclass
class MinimizeActiveVariance(ObjectiveBase):
    """
    Minimize variance relative to a benchmark (tracking error).
    """
    weight: float = 1.0
    
    def get_dependencies(self) -> list[str]:
        return ["get_benchmark_weights"]
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        n = len(context["tickers"])
        cov = context["covariance"]
        benchmark = context["benchmark_weights"]
        
        return ObjectiveExpr(
            linear_coef=np.zeros(n),
            quadratic_matrix=cov,
            gamma=self.weight,
            benchmark_weights=benchmark,
        )


@dataclass
class TargetActiveRisk(ObjectiveBase):
    """
    Maximize alpha while targeting a specific active risk level.
    
    Uses dynamic gamma to achieve the target tracking error.
    """
    target: float = 0.05  # 5% annualized tracking error
    
    def get_dependencies(self) -> list[str]:
        return ["get_benchmark_weights"]
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        alphas = context["alphas"]
        cov = context["covariance"]
        benchmark = context["benchmark_weights"]
        
        return ObjectiveExpr(
            linear_coef=alphas,
            quadratic_matrix=cov,
            gamma=100.0,  # Initial gamma, will be adjusted
            dynamic_gamma=True,
            target_active_risk=self.target,
            benchmark_weights=benchmark,
        )


@dataclass
class RiskParity(ObjectiveBase):
    """
    Equal risk contribution from each asset.
    
    Note: This is a different optimization formulation (not QP).
    """
    
    def get_dependencies(self) -> list[str]:
        return []
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        # Risk parity requires special handling in the optimizer
        # For now, return a placeholder
        raise NotImplementedError("RiskParity requires special optimizer support")


# =============================================================================
# Composite Objective
# =============================================================================

@dataclass
class Objective:
    """
    Composite objective combining multiple terms.
    
    Usage:
        # Weighted sum of terms
        obj = Objective(terms=[
            MaximizeAlpha(weight=1.0),
            MinimizeVariance(weight=10.0),
        ])
        
        # Or with a constraint
        obj = Objective(
            maximize=MaximizeAlpha(),
            subject_to=TargetActiveRisk(target=0.05),
        )
    """
    terms: list[ObjectiveBase] | None = None
    maximize: ObjectiveBase | None = None
    subject_to: ObjectiveBase | None = None
    
    def __post_init__(self):
        if self.terms is not None and (self.maximize is not None or self.subject_to is not None):
            raise ValueError("Cannot specify both 'terms' and 'maximize/subject_to'")
        if self.terms is None and self.maximize is None:
            raise ValueError("Must specify either 'terms' or 'maximize'")
    
    def get_dependencies(self) -> list[str]:
        deps = []
        if self.terms:
            for term in self.terms:
                deps.extend(term.get_dependencies())
        if self.maximize:
            deps.extend(self.maximize.get_dependencies())
        if self.subject_to:
            deps.extend(self.subject_to.get_dependencies())
        return list(set(deps))
    
    def build(self, context: dict[str, Any]) -> ObjectiveExpr:
        if self.terms:
            # Combine terms additively
            n = len(context["tickers"])
            combined_linear = np.zeros(n)
            combined_gamma = 0.0
            cov = context["covariance"]
            
            for term in self.terms:
                expr = term.build(context)
                combined_linear += expr.linear_coef
                combined_gamma += expr.gamma
            
            return ObjectiveExpr(
                linear_coef=combined_linear,
                quadratic_matrix=cov,
                gamma=combined_gamma,
            )
        else:
            # maximize subject_to pattern
            expr = self.maximize.build(context)
            if self.subject_to:
                constraint_expr = self.subject_to.build(context)
                # Merge constraint properties into main expression
                expr.dynamic_gamma = constraint_expr.dynamic_gamma
                expr.target_active_risk = constraint_expr.target_active_risk
                expr.benchmark_weights = constraint_expr.benchmark_weights
            return expr
