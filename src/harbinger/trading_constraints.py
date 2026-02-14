from abc import ABC, abstractmethod
import polars as pl


class TradingConstraint(ABC):
    @abstractmethod
    def apply(self, weights: pl.DataFrame, **kwargs):
        pass


class MinPositionSize(TradingConstraint):
    def __init__(self, dollars: float):
        self.dollars = dollars

    def apply(self, weights: pl.DataFrame, **kwargs):
        prices: pl.DataFrame = kwargs.get('prices')
        return (
            weights
            .join(other=prices, on=['date', 'ticker'], how='left')
            .with_columns(
                pl.col('weight').mul('price').alias('position_size')
            )
            .with_columns(
                pl.when(pl.col('position_size').ge(1))
                .then(pl.col('weight'))
                .otherwise(pl.lit(0))
                .alias('weight')
            )
            .drop('price', 'position_size')
        )
