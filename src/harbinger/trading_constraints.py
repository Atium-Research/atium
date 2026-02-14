"""
Trading constraints.

These are post-optimization heuristics applied before execution.
They modify the optimized weights to satisfy practical trading requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class TradingConstraint(ABC):
    """Base class for trading constraints."""
    
    @abstractmethod
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Apply this constraint to the target weights.
        
        Args:
            target_weights: Weights from optimizer
            current_weights: Current portfolio weights
            context: Dict containing prices, portfolio_value, etc.
            
        Returns:
            Modified weights dict
        """
        pass


# =============================================================================
# Position Size Constraints
# =============================================================================

@dataclass
class MinPositionSize(TradingConstraint):
    """Minimum position size in dollars."""
    
    dollars: float = 1_000
    
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        portfolio_value = context["portfolio_value"]
        
        filtered = {
            ticker: weight
            for ticker, weight in target_weights.items()
            if abs(weight * portfolio_value) >= self.dollars
        }
        
        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}
        
        return filtered


@dataclass
class MinTradeSize(TradingConstraint):
    """Minimum trade size in dollars."""
    
    dollars: float = 1_000
    
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        portfolio_value = context["portfolio_value"]
        
        result = {}
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            trade_value = abs(target - current) * portfolio_value
            
            if trade_value >= self.dollars:
                result[ticker] = target
            else:
                # Keep current weight if trade too small
                result[ticker] = current
        
        return result


# =============================================================================
# Turnover Constraints
# =============================================================================

@dataclass
class MaxTurnover(TradingConstraint):
    """Maximum one-way turnover per rebalance."""
    
    max_turnover: float = 0.25  # 25%
    
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())
        
        # Calculate turnover
        turnover = sum(
            abs(target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2
        
        if turnover <= self.max_turnover:
            return target_weights
        
        # Blend toward target to cap turnover
        blend = self.max_turnover / turnover if turnover > 0 else 1.0
        
        blended = {
            t: current_weights.get(t, 0) + blend * (target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        }
        
        # Clean up near-zeros
        return {k: v for k, v in blended.items() if abs(v) > 1e-10}


# =============================================================================
# Execution Constraints
# =============================================================================

@dataclass
class RoundLots(TradingConstraint):
    """Round positions to lot sizes."""
    
    shares: int = 100
    
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        portfolio_value = context["portfolio_value"]
        prices = context["prices"]
        
        result = {}
        for ticker, weight in target_weights.items():
            if ticker not in prices:
                result[ticker] = weight
                continue
            
            price = prices[ticker]
            target_value = weight * portfolio_value
            target_shares = target_value / price
            
            # Round to lot size
            rounded_shares = round(target_shares / self.shares) * self.shares
            rounded_value = rounded_shares * price
            rounded_weight = rounded_value / portfolio_value
            
            if rounded_weight > 0:
                result[ticker] = rounded_weight
        
        # Renormalize (may not sum to 1 due to rounding)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return result


@dataclass
class MaxPositions(TradingConstraint):
    """Maximum number of positions."""
    
    max_positions: int = 50
    
    def apply(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        if len(target_weights) <= self.max_positions:
            return target_weights
        
        # Keep top N by weight
        sorted_weights = sorted(
            target_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        top_n = dict(sorted_weights[:self.max_positions])
        
        # Renormalize
        total = sum(top_n.values())
        if total > 0:
            top_n = {k: v / total for k, v in top_n.items()}
        
        return top_n
