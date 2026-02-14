from abc import ABC, abstractmethod
import polars as pl

class TradingConstraint(ABC):
    @abstractmethod
    def apply(self, weights: pl.DataFrame, capital: float):
        pass


class MinPositionSize(TradingConstraint):
    def __init__(self, dollars: float):
        self.dollars = dollars

    def apply(self, weights: pl.DataFrame, capital: float):
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
