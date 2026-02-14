from abc import ABC, abstractmethod
import polars as pl


class CostModel(ABC):
    @abstractmethod
    def compute_costs(
        self,
        old_weights: pl.DataFrame | None,
        new_weights: pl.DataFrame,
        capital: float,
    ) -> float:
        pass


class NoCost(CostModel):
    def compute_costs(
        self,
        old_weights: pl.DataFrame | None,
        new_weights: pl.DataFrame,
        capital: float,
    ) -> float:
        return 0.0


class LinearCost(CostModel):
    def __init__(self, bps: float):
        self.bps = bps

    def compute_costs(
        self,
        old_weights: pl.DataFrame | None,
        new_weights: pl.DataFrame,
        capital: float,
    ) -> float:
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
