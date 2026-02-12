"""Vectorized backtesting engine."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from harbinger.metrics import calculate_metrics


@dataclass
class Constraints:
    """Portfolio constraints for backtesting."""
    
    min_position_value: float = 1.0       # Minimum position size in dollars
    min_trade_value: float = 10.0         # Minimum trade size in dollars
    max_position_pct: float = 0.10        # Maximum single position as % of portfolio
    max_turnover: float = 1.0             # Maximum daily turnover as % of portfolio
    allow_short: bool = False             # Allow short positions


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    equity_curve: pl.DataFrame           # date, portfolio_value
    positions: pl.DataFrame              # date, ticker, shares, value, weight
    trades: pl.DataFrame                 # date, ticker, side, shares, price, cost
    daily_returns: pl.Series             # Portfolio daily returns
    metrics: dict                        # Performance metrics
    
    def summary(self) -> str:
        """Print summary of backtest results."""
        m = self.metrics
        return f"""
Backtest Results
================
Total Return:     {m['total_return']*100:>8.2f}%
Annual Return:    {m['annualized_return']*100:>8.2f}%
Annual Vol:       {m['annualized_volatility']*100:>8.2f}%
Sharpe Ratio:     {m['sharpe_ratio']:>8.2f}
Sortino Ratio:    {m['sortino_ratio']:>8.2f}
Max Drawdown:     {m['max_drawdown']*100:>8.2f}%
Calmar Ratio:     {m['calmar_ratio']:>8.2f}
"""


@dataclass
class Backtest:
    """Vectorized backtesting engine."""
    
    initial_capital: float = 100_000.0
    constraints: Constraints = field(default_factory=Constraints)
    slippage_bps: float = 5.0             # Slippage in basis points
    commission_bps: float = 10.0          # Commission in basis points
    
    def run(
        self,
        prices: pl.DataFrame,
        weights: pl.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest with target weights.
        
        Args:
            prices: DataFrame with columns [ticker, date, close]
            weights: DataFrame with columns [ticker, date, weight]
        
        Returns:
            BacktestResult with equity curve, positions, trades, and metrics
        """
        # Get sorted unique dates
        dates = sorted(weights["date"].unique().to_list())
        
        # Initialize tracking
        equity_history = []
        position_history = []
        trade_history = []
        
        cash = self.initial_capital
        holdings: dict[str, float] = {}  # ticker -> shares
        
        for date in dates:
            # Get today's prices
            today_prices = prices.filter(pl.col("date") == date)
            price_map = dict(zip(today_prices["ticker"], today_prices["close"]))
            
            # Calculate current portfolio value
            holdings_value = sum(
                shares * price_map.get(ticker, 0)
                for ticker, shares in holdings.items()
            )
            portfolio_value = cash + holdings_value
            
            # Get target weights for today
            today_weights = weights.filter(pl.col("date") == date)
            target_weights = dict(zip(today_weights["ticker"], today_weights["weight"]))
            
            # Apply constraints and calculate target positions
            target_weights = self._apply_constraints(target_weights, portfolio_value)
            
            # Calculate trades needed
            trades = self._calculate_trades(
                holdings, target_weights, price_map, portfolio_value
            )
            
            # Execute trades
            for ticker, shares_delta in trades.items():
                if ticker not in price_map:
                    continue
                    
                price = price_map[ticker]
                trade_value = abs(shares_delta * price)
                
                # Apply slippage and commission
                slippage_cost = trade_value * (self.slippage_bps / 10000)
                commission_cost = trade_value * (self.commission_bps / 10000)
                total_cost = slippage_cost + commission_cost
                
                if shares_delta > 0:  # Buy
                    cash -= (shares_delta * price + total_cost)
                else:  # Sell
                    cash += (abs(shares_delta) * price - total_cost)
                
                holdings[ticker] = holdings.get(ticker, 0) + shares_delta
                
                # Clean up zero positions
                if abs(holdings[ticker]) < 1e-10:
                    del holdings[ticker]
                
                trade_history.append({
                    "date": date,
                    "ticker": ticker,
                    "side": "buy" if shares_delta > 0 else "sell",
                    "shares": abs(shares_delta),
                    "price": price,
                    "cost": total_cost,
                })
            
            # Record end-of-day state
            holdings_value = sum(
                shares * price_map.get(ticker, 0)
                for ticker, shares in holdings.items()
            )
            portfolio_value = cash + holdings_value
            
            equity_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "invested": holdings_value,
            })
            
            for ticker, shares in holdings.items():
                if ticker in price_map:
                    position_history.append({
                        "date": date,
                        "ticker": ticker,
                        "shares": shares,
                        "value": shares * price_map[ticker],
                        "weight": (shares * price_map[ticker]) / portfolio_value,
                    })
        
        # Build result DataFrames
        equity_curve = pl.DataFrame(equity_history)
        positions = pl.DataFrame(position_history) if position_history else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "shares": pl.Float64, "value": pl.Float64, "weight": pl.Float64}
        )
        trades = pl.DataFrame(trade_history) if trade_history else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "side": pl.Utf8, "shares": pl.Float64, "price": pl.Float64, "cost": pl.Float64}
        )
        
        # Calculate daily returns
        daily_returns = (
            equity_curve["portfolio_value"].diff() / equity_curve["portfolio_value"].shift(1)
        ).drop_nulls()
        
        # Calculate metrics
        metrics = calculate_metrics(
            daily_returns.to_numpy(),
            equity_curve["portfolio_value"].to_numpy()
        )
        
        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions,
            trades=trades,
            daily_returns=daily_returns,
            metrics=metrics,
        )
    
    def _apply_constraints(
        self,
        weights: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, float]:
        """Apply position constraints to target weights."""
        constrained = {}
        
        for ticker, weight in weights.items():
            # Skip short positions if not allowed
            if weight < 0 and not self.constraints.allow_short:
                continue
            
            # Apply max position constraint
            weight = min(weight, self.constraints.max_position_pct)
            weight = max(weight, -self.constraints.max_position_pct)
            
            # Skip positions below minimum value
            if abs(weight * portfolio_value) < self.constraints.min_position_value:
                continue
            
            constrained[ticker] = weight
        
        # Renormalize weights to sum to <= 1
        total = sum(constrained.values())
        if total > 1.0:
            constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
    
    def _calculate_trades(
        self,
        holdings: dict[str, float],
        target_weights: dict[str, float],
        prices: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, float]:
        """Calculate trades needed to reach target weights."""
        trades = {}
        
        all_tickers = set(holdings.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            if ticker not in prices:
                continue
                
            price = prices[ticker]
            current_shares = holdings.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)
            
            target_value = target_weight * portfolio_value
            target_shares = target_value / price if price > 0 else 0
            
            shares_delta = target_shares - current_shares
            trade_value = abs(shares_delta * price)
            
            # Skip tiny trades
            if trade_value < self.constraints.min_trade_value:
                continue
            
            trades[ticker] = shares_delta
        
        # Apply turnover constraint
        total_turnover = sum(abs(s * prices.get(t, 0)) for t, s in trades.items())
        max_turnover_value = self.constraints.max_turnover * portfolio_value
        
        if total_turnover > max_turnover_value and total_turnover > 0:
            scale = max_turnover_value / total_turnover
            trades = {k: v * scale for k, v in trades.items()}
        
        return trades
