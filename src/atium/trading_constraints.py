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
        renormalize: If True, rescale remaining weights to sum to 1.0
    """

    def __init__(self, dollars: float, renormalize: bool = True):
        self.dollars = dollars
        self.renormalize = renormalize

    def apply(self, weights: pl.DataFrame, capital: float):
        """Set weights to zero where weight * capital < self.dollars."""
        result = (
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

        if self.renormalize:
            result = result.with_columns(
                (pl.col('weight') / pl.col('weight').sum().over('date')).alias('weight')
            )

        return result


class MaxPositionCount(TradingConstraint):
    """Keep only the top N positions by absolute weight, zero out the rest.

    Args:
        max_positions: Maximum number of non-zero positions to hold.
        renormalize: If True, rescale remaining weights to sum to 1.0
    """

    def __init__(self, max_positions: int, renormalize: bool = True):
        self.max_positions = max_positions
        self.renormalize = renormalize

    def apply(self, weights: pl.DataFrame, capital: float):
        """Keep only the top max_positions by absolute weight, zero out the rest."""
        result = (
            weights
            .with_columns(
                pl.col('weight').abs().rank(method='ordinal', descending=True).over('date').alias('rank')
            )
            .with_columns(
                pl.when(pl.col('rank') <= self.max_positions)
                .then(pl.col('weight'))
                .otherwise(pl.lit(0))
                .alias('weight')
            )
            .select('date', 'ticker', 'weight')
        )

        if self.renormalize:
            result = result.with_columns(
                (pl.col('weight') / pl.col('weight').sum().over('date')).alias('weight')
            )

        return result
