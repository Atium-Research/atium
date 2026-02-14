"""
Transaction cost models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class SlippageModel(Enum):
    """Slippage model types."""
    
    FIXED = "fixed"           # Fixed bps
    SQRT_IMPACT = "sqrt"      # Square root market impact
    LINEAR_IMPACT = "linear"  # Linear market impact


@dataclass
class TransactionCost:
    """
    Transaction cost model.
    
    Total cost = commission + slippage + borrow_cost (for shorts)
    
    Args:
        commission_bps: Commission in basis points
        slippage_bps: Fixed slippage in basis points (if using FIXED model)
        slippage_model: Type of slippage model
        impact_coef: Coefficient for impact models
        borrow_cost_bps: Annual cost to borrow shares for shorting
    """
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    slippage_model: SlippageModel = SlippageModel.FIXED
    impact_coef: float = 0.1
    borrow_cost_bps: float = 0.0
    
    def calculate(
        self,
        trade_value: float,
        is_short: bool = False,
        adv: float | None = None,
        holding_days: int = 1,
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            trade_value: Absolute value of trade
            is_short: Whether this is a short sale
            adv: Average daily volume (for impact models)
            holding_days: Expected holding period (for borrow costs)
            
        Returns:
            Total cost in dollars
        """
        # Commission
        commission = trade_value * (self.commission_bps / 10_000)
        
        # Slippage
        if self.slippage_model == SlippageModel.FIXED:
            slippage = trade_value * (self.slippage_bps / 10_000)
        elif self.slippage_model == SlippageModel.SQRT_IMPACT:
            if adv and adv > 0:
                participation = trade_value / adv
                slippage = trade_value * self.impact_coef * np.sqrt(participation)
            else:
                slippage = trade_value * (self.slippage_bps / 10_000)
        elif self.slippage_model == SlippageModel.LINEAR_IMPACT:
            if adv and adv > 0:
                participation = trade_value / adv
                slippage = trade_value * self.impact_coef * participation
            else:
                slippage = trade_value * (self.slippage_bps / 10_000)
        else:
            slippage = 0.0
        
        # Borrow cost for shorts
        borrow = 0.0
        if is_short and self.borrow_cost_bps > 0:
            # Annualized borrow cost, prorated for holding period
            annual_cost = trade_value * (self.borrow_cost_bps / 10_000)
            borrow = annual_cost * (holding_days / 365)
        
        return commission + slippage + borrow
    
    @classmethod
    def zero(cls) -> "TransactionCost":
        """No transaction costs (for testing)."""
        return cls(commission_bps=0, slippage_bps=0)
    
    @classmethod
    def realistic(cls) -> "TransactionCost":
        """Realistic institutional costs."""
        return cls(
            commission_bps=1.0,
            slippage_bps=0.0,
            slippage_model=SlippageModel.SQRT_IMPACT,
            impact_coef=0.1,
            borrow_cost_bps=50.0,
        )
