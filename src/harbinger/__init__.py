"""
Harbinger - A flexible portfolio backtesting framework.

Usage:
    from harbinger import (
        Backtester,
        DataAdapter,
        TradingConfig,
        TargetActiveRisk,
        MaximizeAlpha,
        TransactionCost,
    )
    from harbinger.optimizer_constraints import LongOnly, FullyInvested
    from harbinger.trading_constraints import MinTradeSize, MaxTurnover
"""

# Core
from harbinger.backtester import Backtester, BacktestResult, DependencyError
from harbinger.data import DataAdapter
from harbinger.config import TradingConfig

# Objectives
from harbinger.objectives import (
    Objective,
    ObjectiveBase,
    MaximizeAlpha,
    MaximizeSharpe,
    MinimizeVariance,
    MinimizeActiveVariance,
    TargetActiveRisk,
    RiskParity,
)

# Costs
from harbinger.costs import TransactionCost, SlippageModel

__all__ = [
    # Core
    "Backtester",
    "BacktestResult",
    "DataAdapter",
    "TradingConfig",
    "DependencyError",
    # Objectives
    "Objective",
    "ObjectiveBase",
    "MaximizeAlpha",
    "MaximizeSharpe",
    "MinimizeVariance",
    "MinimizeActiveVariance",
    "TargetActiveRisk",
    "RiskParity",
    # Costs
    "TransactionCost",
    "SlippageModel",
]
