"""
Backtester - main entry point for running backtests.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from harbinger.data import DataAdapter
from harbinger.objectives import ObjectiveBase, Objective
from harbinger.optimizer_constraints import OptimizerConstraint
from harbinger.trading_constraints import TradingConstraint
from harbinger.config import TradingConfig
from harbinger.costs import TransactionCost


@dataclass
class BacktestResult:
    """Results from a backtest."""
    
    daily_results: pl.DataFrame
    weights: pl.DataFrame
    trades: pl.DataFrame
    config: dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> dict[str, float]:
        """Calculate summary statistics."""
        if len(self.daily_results) < 2:
            return {}
        
        returns = self.daily_results["return"].drop_nulls()
        equity = self.daily_results["portfolio_value"]
        
        total_return = float(equity[-1] / equity[0] - 1)
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        peak = equity.cum_max()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        
        # Sortino
        downside = returns.filter(returns < 0)
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 1
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0,
            "total_trades": len(self.trades),
            "total_costs": float(self.trades["cost"].sum()) if len(self.trades) > 0 else 0,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        s = self.summary()
        if not s:
            print("No results to summarize.")
            return
        
        print(f"""
Backtest Results
================
Total Return:     {s['total_return']*100:>8.2f}%
Annual Return:    {s['annualized_return']*100:>8.2f}%
Annual Vol:       {s['annualized_volatility']*100:>8.2f}%
Sharpe Ratio:     {s['sharpe_ratio']:>8.2f}
Sortino Ratio:    {s['sortino_ratio']:>8.2f}
Max Drawdown:     {s['max_drawdown']*100:>8.2f}%
Calmar Ratio:     {s['calmar_ratio']:>8.2f}
Total Trades:     {s['total_trades']:>8,}
Total Costs:      ${s['total_costs']:>10,.2f}
""")


class DependencyError(Exception):
    """Raised when a required data method is not implemented."""
    pass


class Backtester:
    """
    Main backtester class.
    
    Usage:
        backtest = Backtester(
            data=MyDataAdapter(db),
            objective=TargetActiveRisk(target=0.05),
            optimizer_constraints=[LongOnly(), FullyInvested()],
            trading=TradingConfig.default(),
            initial_capital=1_000_000,
        )
        result = backtest.run(start_date="2020-01-01", end_date="2024-12-31")
    """
    
    def __init__(
        self,
        data: DataAdapter,
        objective: ObjectiveBase | Objective,
        optimizer_constraints: list[OptimizerConstraint] | None = None,
        trading: TradingConfig | None = None,
        initial_capital: float = 1_000_000,
    ):
        self.data = data
        self.objective = objective
        self.optimizer_constraints = optimizer_constraints or []
        self.trading = trading or TradingConfig.default()
        self.initial_capital = initial_capital
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check that all required data methods are implemented."""
        required_methods = set()
        
        # Collect dependencies from objective
        if hasattr(self.objective, 'get_dependencies'):
            required_methods.update(self.objective.get_dependencies())
        
        # Collect dependencies from optimizer constraints
        for constraint in self.optimizer_constraints:
            if hasattr(constraint, 'get_dependencies'):
                required_methods.update(constraint.get_dependencies())
        
        # Check each required method
        for method_name in required_methods:
            method = getattr(self.data, method_name, None)
            if method is None:
                raise DependencyError(
                    f"Objective/constraint requires {method_name}() but "
                    f"{self.data.__class__.__name__} does not implement it."
                )
            
            # Try to call the method to see if it raises NotImplementedError
            # We can't actually test this without a date, so we just check existence
    
    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            start_date: Start date (inclusive). If None, uses first available date.
            end_date: End date (inclusive). If None, uses last available date.
            
        Returns:
            BacktestResult with daily PnL, weights, and trades
        """
        # Get dates from data adapter
        all_dates = self.data.get_dates()
        
        # Filter to date range
        dates = self._filter_dates(all_dates, start_date, end_date)
        
        if len(dates) < 2:
            raise ValueError("Not enough dates for backtest")
        
        # Initialize state
        portfolio_value = self.initial_capital
        current_weights: dict[str, float] = {}
        
        daily_results = []
        all_weights = []
        all_trades = []
        
        # Main loop
        for i, date in enumerate(tqdm(dates[:-1], desc="Backtesting", unit="day")):
            # Skip if not a rebalance date
            if not self._is_rebalance_date(date, i, dates):
                # Still need to mark-to-market
                returns = self.data.get_returns(date)
                portfolio_return = sum(
                    current_weights.get(t, 0) * returns.get(t, 0)
                    for t in current_weights
                )
                pnl = portfolio_value * portfolio_return
                portfolio_value += pnl
                
                daily_results.append({
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "pnl": pnl,
                    "return": portfolio_return,
                    "turnover": 0.0,
                })
                continue
            
            # Build context for this date
            context = self._build_context(date, portfolio_value, current_weights)
            
            if context is None:
                continue
            
            # Run optimization
            target_weights = self._optimize(context)
            
            # Apply trading constraints
            final_weights = self._apply_trading_constraints(
                target_weights, current_weights, context
            )
            
            # Calculate trades and costs
            trades, day_costs = self._calculate_trades(
                final_weights, current_weights, context
            )
            all_trades.extend(trades)
            
            # Record weights
            for ticker, weight in final_weights.items():
                all_weights.append({
                    "date": date,
                    "ticker": ticker,
                    "weight": weight,
                })
            
            # Calculate portfolio return
            returns = self.data.get_returns(date)
            portfolio_return = sum(
                final_weights.get(t, 0) * returns.get(t, 0)
                for t in final_weights
            )
            
            # Update portfolio value
            pnl = portfolio_value * portfolio_return - day_costs
            portfolio_value += pnl
            
            # Calculate turnover
            all_tickers = set(final_weights.keys()) | set(current_weights.keys())
            turnover = sum(
                abs(final_weights.get(t, 0) - current_weights.get(t, 0))
                for t in all_tickers
            ) / 2
            
            daily_results.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "pnl": pnl,
                "return": portfolio_return - day_costs / (portfolio_value - pnl) if portfolio_value != pnl else 0,
                "turnover": turnover,
            })
            
            current_weights = final_weights
        
        return BacktestResult(
            daily_results=pl.DataFrame(daily_results),
            weights=pl.DataFrame(all_weights) if all_weights else pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.Utf8, "weight": pl.Float64}
            ),
            trades=pl.DataFrame(all_trades) if all_trades else pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.Utf8, "old_weight": pl.Float64,
                        "new_weight": pl.Float64, "value": pl.Float64, "cost": pl.Float64}
            ),
            config={
                "initial_capital": self.initial_capital,
                "objective": str(self.objective),
                "optimizer_constraints": [str(c) for c in self.optimizer_constraints],
            },
        )
    
    def _filter_dates(self, all_dates, start_date, end_date):
        """Filter dates to the specified range."""
        # TODO: Implement date filtering
        return all_dates
    
    def _is_rebalance_date(self, date, index, dates) -> bool:
        """Check if this is a rebalance date based on frequency."""
        freq = self.trading.rebalance_frequency
        
        if freq == "daily":
            return True
        elif freq == "weekly":
            # Rebalance on Mondays (or first day of week)
            return index == 0 or index % 5 == 0
        elif freq == "monthly":
            # Rebalance on first day of month
            if index == 0:
                return True
            prev_date = dates[index - 1]
            return date.month != prev_date.month
        elif isinstance(freq, list):
            return date in freq
        
        return True
    
    def _build_context(self, date, portfolio_value, current_weights) -> dict | None:
        """Build optimization context for a date."""
        try:
            universe = self.data.get_universe(date)
            alphas_dict = self.data.get_alphas(date)
            cov, cov_tickers = self.data.get_covariance(date)
            prices = self.data.get_prices(date)
            
            if not universe or len(cov_tickers) == 0:
                return None
            
            # Filter to common tickers
            tickers = sorted(set(universe) & set(cov_tickers) & set(alphas_dict.keys()))
            
            if not tickers:
                return None
            
            # Build arrays in ticker order
            alphas = np.array([alphas_dict.get(t, 0) for t in tickers])
            
            context = {
                "date": date,
                "tickers": tickers,
                "alphas": alphas,
                "covariance": cov,
                "prices": prices,
                "portfolio_value": portfolio_value,
                "current_weights": current_weights,
            }
            
            # Add optional data if available
            try:
                context["benchmark_weights"] = np.array([
                    self.data.get_benchmark_weights(date).get(t, 0)
                    for t in tickers
                ])
            except NotImplementedError:
                pass
            
            try:
                context["sector_mapping"] = self.data.get_sector_mapping(date)
            except NotImplementedError:
                pass
            
            try:
                context["factor_exposures"] = self.data.get_factor_exposures(date)
            except NotImplementedError:
                pass
            
            return context
            
        except Exception as e:
            # Log and skip this date
            print(f"Warning: Error building context for {date}: {e}")
            return None
    
    def _optimize(self, context: dict) -> dict[str, float]:
        """Run the optimization."""
        # TODO: Implement actual optimization using cvxpy
        # For now, return equal weights as placeholder
        tickers = context["tickers"]
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}
    
    def _apply_trading_constraints(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict,
    ) -> dict[str, float]:
        """Apply trading constraints in order."""
        weights = target_weights
        
        for constraint in self.trading.constraints:
            weights = constraint.apply(weights, current_weights, context)
        
        return weights
    
    def _calculate_trades(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        context: dict,
    ) -> tuple[list[dict], float]:
        """Calculate trades and costs."""
        portfolio_value = context["portfolio_value"]
        prices = context["prices"]
        date = context["date"]
        
        trades = []
        total_cost = 0.0
        
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())
        
        for ticker in all_tickers:
            old_weight = current_weights.get(ticker, 0)
            new_weight = target_weights.get(ticker, 0)
            
            if abs(new_weight - old_weight) > 1e-10 and ticker in prices:
                trade_value = abs(new_weight - old_weight) * portfolio_value
                is_short = new_weight < 0
                cost = self.trading.costs.calculate(trade_value, is_short=is_short)
                total_cost += cost
                
                trades.append({
                    "date": date,
                    "ticker": ticker,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "value": trade_value,
                    "cost": cost,
                })
        
        return trades, total_cost
