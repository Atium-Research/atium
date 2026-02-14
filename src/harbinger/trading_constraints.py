from abc import ABC, abstractmethod
import polars as pl

class TradingConstraint(ABC):
    """Base class for post-optimization filters applied to portfolio weights."""

    @abstractmethod
    def apply(self, weights: pl.DataFrame, capital: float):
        """Apply this constraint to the weights DataFrame and return the filtered result."""
        pass


class MinPositionSize(TradingConstraint):
    """Zero out positions whose dollar value falls below a minimum threshold.

    Args:
        dollars: Minimum position size in dollars. Positions smaller than
            this are set to zero weight.
    """

    def __init__(self, dollars: float):
        self.dollars = dollars

    def apply(self, weights: pl.DataFrame, capital: float):
        """Set weights to zero where weight * capital < self.dollars."""
        return (
            weights
            .with_columns(
                pl.col('weight').mul(capital).alias('position_size')
            )
            .with_columns(
                pl.when(pl.col('position_size').ge(self.dollars))
                .then(pl.col('weight'))
                .otherwise(pl.lit(0))
                .alias('weight')
            )
            .select('date', 'ticker', 'weight')
        )
