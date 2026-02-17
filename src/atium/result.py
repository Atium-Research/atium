import polars as pl
from atium.models import (
    PositionResults,
    PortfolioReturns,
    BenchmarkReturns,
    ActiveReturns,
    PerformanceSummary,
)


class BacktestResult:
    """Container for backtest output with performance analytics.

    Wraps the per-position results DataFrame and provides methods for
    computing portfolio-level return series and summary statistics.

    The underlying DataFrame has columns [date, ticker, weight, value, return, pnl].
    """

    def __init__(
        self,
        results: PositionResults,
        benchmark_returns: BenchmarkReturns | None = None,
    ):
        self.results = results
        self.benchmark_returns = benchmark_returns

    def portfolio_returns(self) -> PortfolioReturns:
        """Aggregate position-level results into daily portfolio returns.

        Returns a DataFrame with columns [date, portfolio_value, portfolio_return].
        """
        return PortfolioReturns(
            self.results
            .group_by('date')
            .agg(
                pl.col('value').add(pl.col('pnl')).sum().alias('portfolio_value'),
                pl.col('return').mul(pl.col('weight')).sum().alias('portfolio_return'),
            )
            .sort('date')
        )

    def sharpe_ratio(self, annualization_factor: float = 252) -> float:
        """Compute the annualized Sharpe ratio (mean / std * sqrt(factor))."""
        returns = self.portfolio_returns()['portfolio_return']
        mean = returns.mean()
        std = returns.std()
        if std == 0 or std is None or mean is None:
            return 0.0
        return float(mean / std * (annualization_factor ** 0.5))

    def annualized_return(self) -> float:
        """Compute the annualized return as a percentage."""
        returns = self.portfolio_returns()['portfolio_return']
        mean = returns.mean()
        if mean is None:
            return 0.0
        return float(mean * 252 * 100)

    def annualized_volatility(self) -> float:
        """Compute the annualized volatility as a percentage."""
        returns = self.portfolio_returns()['portfolio_return']
        std = returns.std()
        if std is None:
            return 0.0
        return float(std * (252 ** 0.5) * 100)

    def active_returns(self) -> ActiveReturns:
        """Compute daily active returns (portfolio minus benchmark).

        Returns a DataFrame with columns [date, active_return].
        Raises ValueError if no benchmark was provided.
        """
        if self.benchmark_returns is None:
            raise ValueError("No benchmark returns available.")
        return ActiveReturns(
            self.portfolio_returns()
            .join(self.benchmark_returns, on='date', how='inner')
            .with_columns(
                (pl.col('portfolio_return') - pl.col('benchmark_return'))
                .alias('active_return'),
            )
            .select('date', 'active_return')
        )

    def active_return_annualized(self) -> float:
        """Compute the annualized active return as a percentage."""
        mean = self.active_returns()['active_return'].mean()
        if mean is None:
            return 0.0
        return float(mean * 252 * 100)

    def tracking_error(self) -> float:
        """Compute the annualized tracking error as a percentage."""
        std = self.active_returns()['active_return'].std()
        if std is None:
            return 0.0
        return float(std * (252 ** 0.5) * 100)

    def information_ratio(self) -> float:
        """Compute the information ratio (annualized active return / tracking error)."""
        te = self.tracking_error()
        if te == 0:
            return 0.0
        return self.active_return_annualized() / te

    def relative_max_drawdown(self) -> float:
        """Compute the maximum drawdown of cumulative active returns as a decimal."""
        active = self.active_returns()
        cumulative = (1 + active['active_return']).cum_prod()
        running_max = cumulative.cum_max()
        drawdown = (cumulative - running_max) / running_max
        dd = drawdown.min()
        if dd is None:
            return 0.0
        return float(dd)

    def max_drawdown(self) -> float:
        """Compute the maximum peak-to-trough drawdown as a decimal (e.g. -0.10 for 10%)."""
        portfolio = self.portfolio_returns()
        cumulative = portfolio['portfolio_value']
        running_max = cumulative.cum_max()
        drawdown = (cumulative - running_max) / running_max
        dd = drawdown.min()
        if dd is None:
            return 0.0
        return float(dd)

    def summary(self) -> PerformanceSummary:
        """Return a single-row DataFrame with key performance metrics.

        When benchmark returns are available, includes active return,
        tracking error, information ratio, and relative max drawdown.
        """
        data: dict[str, list] = {
            'annualized_return_pct': [self.annualized_return()],
            'annualized_volatility_pct': [self.annualized_volatility()],
            'sharpe_ratio': [self.sharpe_ratio()],
            'max_drawdown_pct': [self.max_drawdown() * 100],
            'active_return_pct': [self.active_return_annualized() if self.benchmark_returns is not None else None],
            'tracking_error_pct': [self.tracking_error() if self.benchmark_returns is not None else None],
            'information_ratio': [self.information_ratio() if self.benchmark_returns is not None else None],
            'relative_max_drawdown_pct': [self.relative_max_drawdown() * 100 if self.benchmark_returns is not None else None],
        }
        return PerformanceSummary(pl.DataFrame(data))

    def plot_equity_curve(self, path: str = 'equity_curve.png') -> str:
        """Save an equity curve chart to disk and return the file path.

        Requires matplotlib and seaborn (dev dependencies).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        portfolio = self.portfolio_returns()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=portfolio, x='date', y='portfolio_value', ax=ax)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path
