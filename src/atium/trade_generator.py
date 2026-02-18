from atium.trading_constraints import TradingConstraint
from atium.types import PortfolioWeights


class TradeGenerator:
    """Applies trading constraints to portfolio weights sequentially.

    Args:
        constraints: Trading constraints to apply in order.
    """

    def __init__(self, constraints: list[TradingConstraint]):
        self.constraints = constraints

    def apply(self, weights: PortfolioWeights, capital: float) -> PortfolioWeights:
        """Apply each trading constraint to the weights and return the result."""
        for constraint in self.constraints:
            weights = constraint.apply(weights, capital=capital)
        return weights
