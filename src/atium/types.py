from __future__ import annotations

import datetime as dt
import numpy as np
import polars as pl


class Alphas:
    """Alpha signals for a set of assets on a given date.

    Wraps a DataFrame with columns [date, ticker, alpha], sorted by ticker.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df.sort('ticker')

    @property
    def tickers(self) -> list[str]:
        return self._df['ticker'].to_list()

    @property
    def date(self) -> dt.date:
        return self._df['date'][0]

    def to_numpy(self) -> np.ndarray:
        return self._df['alpha'].to_numpy()

    def to_frame(self) -> pl.DataFrame:
        return self._df


class BenchmarkWeights:
    """Benchmark portfolio weights for a set of assets on a given date.

    Wraps a DataFrame with columns [date, ticker, weight], sorted by ticker.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df.sort('ticker')

    @property
    def tickers(self) -> list[str]:
        return self._df['ticker'].to_list()

    @property
    def date(self) -> dt.date:
        return self._df['date'][0]

    def align_to(self, tickers: list[str]) -> BenchmarkWeights:
        """Return new BenchmarkWeights filtered and aligned to the given tickers.

        Tickers not present in the benchmark receive a weight of 0.
        """
        aligned = (
            pl.DataFrame({'ticker': tickers})
            .join(self._df.select('ticker', 'weight'), on='ticker', how='left')
            .with_columns(pl.col('weight').fill_null(0.0))
            .sort('ticker')
            .with_columns(pl.lit(self._df['date'][0]).alias('date'))
        )
        return BenchmarkWeights(aligned)

    def to_numpy(self) -> np.ndarray:
        return self._df['weight'].to_numpy()

    def to_frame(self) -> pl.DataFrame:
        return self._df
