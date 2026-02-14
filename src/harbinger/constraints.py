"""
Constraint classes for portfolio optimization and trading.

Two types:
- OptimizerConstraint: Applied after MVO optimization
- TradingConstraint: Applied before trade execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


# =============================================================================
# Base Classes
# =============================================================================

class Constraint(ABC):
    """Base constraint class."""
    
    @abstractmethod
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        """Apply constraint to weights. Returns modified weights."""
        pass


class OptimizerConstraint(Constraint):
    """Constraints applied after portfolio optimization."""
    pass


class TradingConstraint(Constraint):
    """Constraints applied before trade execution."""
    pass


# =============================================================================
# Optimizer Constraints
# =============================================================================

@dataclass
class LongOnly(OptimizerConstraint):
    """No short positions allowed."""
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        return {k: max(v, 0) for k, v in weights.items()}


@dataclass
class MaxWeight(OptimizerConstraint):
    """Maximum weight per position."""
    
    max_weight: float = 0.10
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        return {k: min(v, self.max_weight) for k, v in weights.items()}


@dataclass 
class MinWeight(OptimizerConstraint):
    """Minimum weight per position (for shorts)."""
    
    min_weight: float = -0.10
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        return {k: max(v, self.min_weight) for k, v in weights.items()}


@dataclass
class FullyInvested(OptimizerConstraint):
    """Weights must sum to 1."""
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights


# =============================================================================
# Trading Constraints
# =============================================================================

@dataclass
class MinPositionValue(TradingConstraint):
    """Minimum dollar value per position."""
    
    min_value: float = 1.0
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        portfolio_value = context.get("portfolio_value", 100_000)
        
        filtered = {
            k: v for k, v in weights.items()
            if abs(v * portfolio_value) >= self.min_value
        }
        
        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}
        
        return filtered


@dataclass
class MinTradeValue(TradingConstraint):
    """Minimum dollar value per trade."""
    
    min_value: float = 10.0
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        portfolio_value = context.get("portfolio_value", 100_000)
        current_weights = context.get("current_weights", {})
        
        filtered = {}
        for ticker, weight in weights.items():
            current = current_weights.get(ticker, 0)
            trade_value = abs(weight - current) * portfolio_value
            
            if trade_value >= self.min_value or weight == 0:
                filtered[ticker] = weight
            else:
                filtered[ticker] = current
        
        return filtered


@dataclass
class MaxTurnover(TradingConstraint):
    """Maximum one-way turnover per rebalance."""
    
    max_turnover: float = 1.0
    
    def apply(self, weights: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        current_weights = context.get("current_weights", {})
        all_tickers = set(weights.keys()) | set(current_weights.keys())
        
        # Calculate turnover
        turnover = sum(
            abs(weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2
        
        if turnover <= self.max_turnover:
            return weights
        
        # Blend toward target to cap turnover
        blend = self.max_turnover / turnover if turnover > 0 else 1.0
        
        blended = {
            t: current_weights.get(t, 0) + blend * (weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        }
        
        return {k: v for k, v in blended.items() if abs(v) > 1e-10}


# =============================================================================
# Utility Functions
# =============================================================================

def apply_constraints(
    weights: dict[str, float],
    constraints: list[Constraint],
    context: dict[str, Any],
) -> dict[str, float]:
    """Apply a list of constraints in order."""
    for constraint in constraints:
        weights = constraint.apply(weights, context)
    return weights


def apply_optimizer_constraints(
    weights: dict[str, float],
    constraints: list[Constraint],
    context: dict[str, Any],
) -> dict[str, float]:
    """Apply only OptimizerConstraint instances."""
    optimizer_constraints = [c for c in constraints if isinstance(c, OptimizerConstraint)]
    return apply_constraints(weights, optimizer_constraints, context)


def apply_trading_constraints(
    weights: dict[str, float],
    constraints: list[Constraint],
    context: dict[str, Any],
) -> dict[str, float]:
    """Apply only TradingConstraint instances."""
    trading_constraints = [c for c in constraints if isinstance(c, TradingConstraint)]
    return apply_constraints(weights, trading_constraints, context)
