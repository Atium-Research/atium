#!/usr/bin/env python3
"""
Harbinger Demo — Clean backtester API.

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
    # 2. Configure and run backtest
    # ========================================
    print("\n🚀 Running backtest...")

    config = BacktestConfig(
        initial_capital=100_000,
        target_active_risk=0.05,
        risk_aversion=1.0,
        min_position_value=1.0,
        max_position_weight=0.10,
        max_turnover=1.0,
        long_only=True,
        slippage_bps=5.0,
        commission_bps=10.0,
    )

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
    # 3. Show results
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
