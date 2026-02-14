from abc import ABC, abstractmethod
import datetime as dt
import polars as pl


class MarketDataProvider(ABC):
    """Interface for market data: calendar, universe, prices, and returns."""

    @abstractmethod
    def get_calendar(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """Return the list of trading dates between start and end (inclusive)."""
        pass

    @abstractmethod
    def get_universe(self, date_: dt.date) -> list[str]:
        """Return ticker symbols in the investable universe on the given date."""
        pass

    @abstractmethod
    def get_prices(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, price]."""
        pass

    @abstractmethod
    def get_forward_returns(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, return] containing next-period returns."""
        pass


class AlphaProvider(ABC):
    """Interface for alpha-generation data: signals, scores, and expected returns."""

    @abstractmethod
    def get_signals(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, signal]."""
        pass

    @abstractmethod
    def get_scores(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, score]."""
        pass

    @abstractmethod
    def get_alphas(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, alpha] containing expected returns."""
        pass


class RiskDataProvider(ABC):
    """Interface for risk model data: benchmark, betas, factors, and idiosyncratic volatility."""

    @abstractmethod
    def get_benchmark_weights(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, weight] for the benchmark."""
        pass

    @abstractmethod
    def get_betas(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, beta]."""
        pass

    @abstractmethod
    def get_factor_loadings(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, factor, loading]."""
        pass

    @abstractmethod
    def get_factor_covariances(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, factor_1, factor_2, covariance]."""
        pass

    @abstractmethod
    def get_idio_vol(self, date_: dt.date) -> pl.DataFrame:
        """Return a DataFrame with columns [date, ticker, idio_vol]."""
        pass


class DataAdapter(MarketDataProvider, AlphaProvider, RiskDataProvider):
    """Convenience base that implements all three provider interfaces in a single class."""
    pass
