import datetime as dt
from typing import Protocol

from atium.types import (Alphas, BenchmarkWeights, Betas, FactorCovariances,
                         FactorLoadings, IdioVol, Returns)


class CalendarProvider(Protocol):
    """Provides the list of trading dates for a given range."""

    def get(self, start: dt.date, end: dt.date) -> list[dt.date]: ...


class ReturnsProvider(Protocol):
    """Provides next-period forward returns with columns [date, ticker, return]."""

    def get(self, date_: dt.date) -> Returns: ...


class AlphaProvider(Protocol):
    """Provides expected returns with columns [date, ticker, alpha]."""

    def get(self, date_: dt.date) -> Alphas: ...


class FactorLoadingsProvider(Protocol):
    """Provides factor exposures with columns [date, ticker, factor, loading]."""

    def get(self, date_: dt.date) -> FactorLoadings: ...


class FactorCovariancesProvider(Protocol):
    """Provides factor covariance matrix with columns [date, factor_1, factor_2, covariance]."""

    def get(self, date_: dt.date) -> FactorCovariances: ...


class IdioVolProvider(Protocol):
    """Provides idiosyncratic volatility with columns [date, ticker, idio_vol]."""

    def get(self, date_: dt.date) -> IdioVol: ...


class BetaProvider(Protocol):
    """Provides forecasted asset betas with columns [date, ticker, beta]."""

    def get(self, date_: dt.date) -> Betas: ...


class BenchmarkWeightsProvider(Protocol):
    """Provides benchmark portfolio weights with columns [date, ticker, weight]."""

    def get(self, date_: dt.date) -> BenchmarkWeights: ...
