#!/usr/bin/env python3
"""
Harbinger Demo — Clean backtester API with class-based constraints.

Usage:
    uv run python demo.py
"""

from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from harbinger.data import (
    get_bear_lake_client,
    load_prices,
    load_signals,
    load_returns,
    load_factor_loadings,
    load_factor_covariances,
    load_idio_vol,
    load_benchmark_weights,
)
from harbinger.backtest import backtest, BacktestConfig
from harbinger.constraints import (
    LongOnly,
    FullyInvested,
    MinPositionValue,
    MaxTurnover,
)


def main():
    print("🔮 Harbinger Demo\n")
    print("=" * 60)

    # ========================================
    # 1. Load data
    # ========================================
    print("\n📊 Loading data from Bear Lake...")

    client = get_bear_lake_client()

    start_date = "2023-01-01"
    end_date = "2026-01-31"

    signals = load_signals(client, start_date, end_date, signal_name="reversal")
    print(f"   Signals: {signals.shape}")

    returns = load_returns(client, start_date, end_date)
    print(f"   Returns: {returns.shape}")

    prices = load_prices(client, start_date, end_date)
    print(f"   Prices: {prices.shape}")

    factor_loadings = load_factor_loadings(client, start_date, end_date)
    print(f"   Factor loadings: {factor_loadings.shape}")

    factor_covariances = load_factor_covariances(client, start_date, end_date)
    print(f"   Factor covariances: {factor_covariances.shape}")

    idio_vol = load_idio_vol(client, start_date, end_date)
    print(f"   Idio vol: {idio_vol.shape}")

    benchmark_weights = load_benchmark_weights(client, start_date, end_date)
    print(f"   Benchmark weights: {benchmark_weights.shape}")

    # ========================================
    # 2. Configure backtest
    # ========================================
    print("\n⚙️  Configuring backtest...")

    # Option 1: Dynamic gamma with target active risk (default)
    config = BacktestConfig(
        initial_capital=100_000,
        
        # Risk control: dynamic gamma to achieve 5% active risk
        gamma="dynamic",
        target_active_risk=0.05,
        
        # Optimizer constraints (applied after MVO)
        optimizer_constraints=[
            LongOnly(),
            FullyInvested(),
        ],
        
        # Trading constraints (applied after optimization)
        trading_constraints=[
            MinPositionValue(min_value=1.0),
            MaxTurnover(max_turnover=1.0),
        ],
        
        slippage_bps=5.0,
        commission_bps=10.0,
    )
    
    # Option 2: Fixed gamma (uncomment to use)
    # config = BacktestConfig.with_fixed_gamma(gamma=100.0)

    if config.gamma == "dynamic":
        print(f"   Mode: Dynamic gamma (target active risk: {config.target_active_risk*100:.1f}%)")
    else:
        print(f"   Mode: Fixed gamma = {config.gamma}")
    
    print("   Optimizer constraints:")
    for c in config.optimizer_constraints:
        print(f"      - {c.__class__.__name__}")
    
    print("   Trading constraints:")
    for c in config.trading_constraints:
        print(f"      - {c.__class__.__name__}")

    # ========================================
    # 3. Run backtest
    # ========================================
    print("\n🚀 Running backtest...")

    result = backtest(
        signals=signals,
        returns=returns,
        factor_loadings=factor_loadings,
        factor_covariances=factor_covariances,
        idio_vol=idio_vol,
        benchmark_weights=benchmark_weights,
        prices=prices,
        config=config,
    )

    # ========================================
    # 4. Show results
    # ========================================
    result.print_summary()

    print("📈 Daily Results (last 10 days):")
    print(result.daily_results.tail(10))

    print(f"\n📊 Weights shape: {result.weights.shape}")
    print(f"📊 Trades shape: {result.trades.shape}")

    # Show sample weights
    if not result.weights.is_empty():
        last_date = result.weights["date"].max()
        final_weights = (
            result.weights
            .filter(pl.col("date") == last_date)
            .sort("weight", descending=True)
            .head(10)
        )
        print("\n🏆 Final positions (top 10):")
        for row in final_weights.iter_rows(named=True):
            print(f"   {row['ticker']:>6}: {row['weight']*100:>6.2f}%")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
