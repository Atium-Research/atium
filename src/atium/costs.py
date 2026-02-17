from abc import ABC, abstractmethod
import polars as pl
from atium.models import PortfolioWeights


class CostModel(ABC):
    """Base class for transaction cost estimation."""

    @abstractmethod
    def compute_costs(
        self,
        old_weights: PortfolioWeights | None,
        new_weights: PortfolioWeights,
        capital: float,
    ) -> float:
        """Estimate the dollar cost of rebalancing from old_weights to new_weights.

        Args:
            old_weights: Previous portfolio weights, or None for the first rebalance.
            new_weights: Target portfolio weights with columns [ticker, weight].
            capital: Current portfolio capital in dollars.
        """
        pass


class NoCost(CostModel):
    """Cost model that charges zero transaction costs."""

    def compute_costs(
        self,
        old_weights: PortfolioWeights | None,
        new_weights: PortfolioWeights,
        capital: float,
    ) -> float:
        """Return 0.0 (no costs)."""
        return 0.0


class LinearCost(CostModel):
    """Transaction cost proportional to portfolio turnover.

    Cost is computed as: turnover * capital * bps / 10,000.

    Args:
        bps: Cost in basis points per unit of turnover.
    """

    def __init__(self, bps: float):
        self.bps = bps

    def compute_costs(
        self,
        old_weights: PortfolioWeights | None,
        new_weights: PortfolioWeights,
        capital: float,
    ) -> float:
        """Compute linear transaction costs based on weight turnover."""
        if old_weights is None:
            turnover = new_weights['weight'].abs().sum()
        else:
            combined = (
                new_weights.select('ticker', pl.col('weight').alias('new_weight'))
                .join(
                    old_weights.select('ticker', pl.col('weight').alias('old_weight')),
                    on='ticker',
                    how='full',
                    coalesce=True,
                )
                .fill_null(0)
            )
            turnover = (combined['new_weight'] - combined['old_weight']).abs().sum()

        return turnover * capital * self.bps / 10_000
