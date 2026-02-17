import polars as pl
from atium.trading_constraints import TradingConstraint


class TradeGenerator:
    """Applies trading constraints to portfolio weights sequentially.

    Args:
        constraints: Trading constraints to apply in order.
    """

    def __init__(self, constraints: list[TradingConstraint]):
        self.constraints = constraints

    def apply(self, weights: pl.DataFrame, capital: float) -> pl.DataFrame:
        """Apply each trading constraint to the weights and return the result."""
        for constraint in self.constraints:
            weights = constraint.apply(weights, capital=capital)
        return weights
