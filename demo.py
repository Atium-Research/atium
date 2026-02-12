#!/usr/bin/env python3
"""
Harbinger Demo — Full backtest workflow with real data.

Usage:
    uv run python demo.py
"""

from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

from harbinger.data import get_bear_lake_client, load_prices, load_returns, load_universe
from harbinger.optimize import generate_weights_series
from harbinger.backtest import Backtest, Constraints


def main():
    print("🔮 Harbinger Demo\n")
    print("=" * 60)
    
    # ========================================
    # 1. Load data from Bear Lake
    # ========================================
    print("\n📊 Loading data from Bear Lake...")
    
    client = get_bear_lake_client()
    
    # Date range
    start_date = "2025-01-01"
    end_date = "2026-01-31"
    
    # Load universe
    universe = load_universe(client, "2025-01-02")
    print(f"   Universe: {len(universe)} tickers")
    
    # Load prices and returns
    prices = load_prices(client, start_date, end_date, tickers=universe)
    returns = load_returns(client, start_date, end_date, tickers=universe)
    
    print(f"   Prices: {prices.shape} (tickers × days)")
    print(f"   Returns: {returns.shape}")
    print(f"   Date range: {prices['date'].min()} to {prices['date'].max()}")
    
    # ========================================
    # 2. Generate portfolio weights
    # ========================================
    print("\n⚖️  Generating portfolio weights...")
    
    # Mean-variance optimization with 60-day lookback, rebalance weekly
    weights = generate_weights_series(
        returns,
        method="mean_variance",
        lookback=60,
        rebalance_freq=5,  # Weekly
        risk_aversion=2.0,
    )
    
    print(f"   Generated weights for {weights['date'].n_unique()} rebalance dates")
    print(f"   Average positions per day: {len(weights) / weights['date'].n_unique():.0f}")
    
    # Show sample weights
    print("\n   Sample weights (first rebalance):")
    first_date = weights["date"].min()
    sample = weights.filter(pl.col("date") == first_date).sort("weight", descending=True).head(10)
    for row in sample.iter_rows(named=True):
        print(f"      {row['ticker']:>6}: {row['weight']*100:>5.2f}%")
    
    # ========================================
    # 3. Run backtest
    # ========================================
    print("\n🚀 Running backtest...")
    
    constraints = Constraints(
        min_position_value=100.0,    # Min $100 per position
        min_trade_value=50.0,        # Min $50 per trade
        max_position_pct=0.05,       # Max 5% per position
        max_turnover=0.20,           # Max 20% turnover per day
    )
    
    bt = Backtest(
        initial_capital=100_000,
        constraints=constraints,
        slippage_bps=5.0,
        commission_bps=10.0,
    )
    
    result = bt.run(prices=prices, weights=weights)
    
    # ========================================
    # 4. Show results
    # ========================================
    print(result.summary())
    
    print("📈 Equity Curve:")
    print(result.equity_curve.head(10))
    
    print(f"\n📊 Total trades: {len(result.trades)}")
    print(f"   Trading costs: ${result.trades['cost'].sum():,.2f}")
    
    print("\n🏆 Final positions:")
    final_date = result.positions["date"].max()
    final_positions = (
        result.positions
        .filter(pl.col("date") == final_date)
        .sort("weight", descending=True)
        .head(10)
    )
    for row in final_positions.iter_rows(named=True):
        print(f"   {row['ticker']:>6}: ${row['value']:>10,.2f} ({row['weight']*100:>5.2f}%)")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
