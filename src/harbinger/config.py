"""
Configuration classes.
"""

from dataclasses import dataclass, field
from typing import Literal

from harbinger.trading_constraints import TradingConstraint, MinTradeSize, MaxTurnover
from harbinger.costs import TransactionCost


@dataclass
class TradingConfig:
    """
    Trading configuration.
    
    Defines post-optimization constraints, transaction costs,
    and rebalancing schedule.
    """
    
    constraints: list[TradingConstraint] = field(default_factory=list)
    costs: TransactionCost = field(default_factory=TransactionCost)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] | list = "daily"
    
    @classmethod
    def default(cls) -> "TradingConfig":
        """Sensible defaults for institutional trading."""
        return cls(
            constraints=[
                MinTradeSize(dollars=1_000),
                MaxTurnover(max_turnover=0.25),
            ],
            costs=TransactionCost(
                commission_bps=1.0,
                slippage_bps=5.0,
            ),
            rebalance_frequency="daily",
        )
    
    @classmethod
    def low_cost(cls) -> "TradingConfig":
        """Low-cost configuration (e.g., for index funds)."""
        return cls(
            constraints=[
                MinTradeSize(dollars=10_000),
                MaxTurnover(max_turnover=0.10),
            ],
            costs=TransactionCost(
                commission_bps=0.5,
                slippage_bps=2.0,
            ),
            rebalance_frequency="monthly",
        )
