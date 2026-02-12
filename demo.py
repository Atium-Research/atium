#!/usr/bin/env python3
"""
Harbinger Demo — MVO backtest with reversal signal and factor model covariance.

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
    load_factor_loadings,
    load_factor_covariances,
    load_idio_vol,
)
from harbinger.optimize import (
    build_factor_covariance,
    mean_variance_optimize,
    signal_to_expected_returns,
)
from harbinger.backtest import Backtest, Constraints


def main():
    print("🔮 Harbinger Demo — Reversal + Factor Model MVO\n")
    print("=" * 60)

    # ========================================
    # 1. Load ALL data upfront (batch)
    # ========================================
    print("\n📊 Loading data from Bear Lake...")

    client = get_bear_lake_client()

    start_date = "2025-06-01"
    end_date = "2026-01-31"

    # Load all data in batch
    prices = load_prices(client, start_date, end_date)
    print(f"   Prices: {prices.shape}")

    signals = load_signals(client, start_date, end_date, signal_name="reversal")
    print(f"   Signals: {signals.shape}")

    factor_loadings = load_factor_loadings(client, start_date, end_date)
    print(f"   Factor loadings: {factor_loadings.shape}")

    factor_covariances = load_factor_covariances(client, start_date, end_date)
    print(f"   Factor covariances: {factor_covariances.shape}")

    idio_vol = load_idio_vol(client, start_date, end_date)
    print(f"   Idio vol: {idio_vol.shape}")

    # Get trading dates (dates with all required data)
    signal_dates = set(signals["date"].unique().to_list())
    factor_dates = set(factor_loadings["date"].unique().to_list())
    idio_dates = set(idio_vol["date"].unique().to_list())
    trading_dates = sorted(signal_dates & factor_dates & idio_dates)
    print(f"   Trading dates: {len(trading_dates)}")

    # ========================================
    # 2. Generate daily weights using MVO
    # ========================================
    print("\n⚖️  Running daily MVO with factor model covariance...")

    all_weights = []

    for i, date in enumerate(trading_dates):
        if i % 20 == 0:
            print(f"   Processing date {i+1}/{len(trading_dates)}: {date}")

        # Build covariance matrix for this date
        cov_matrix, tickers = build_factor_covariance(
            factor_loadings, factor_covariances, idio_vol, date=date
        )

        if len(tickers) == 0:
            continue

        # Get expected returns from reversal signal
        expected_returns = signal_to_expected_returns(signals, date, scale=1.0)
        expected_returns = {k: v for k, v in expected_returns.items() if k in tickers}

        if not expected_returns:
            continue

        # Run MVO (long only)
        weights = mean_variance_optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            tickers=tickers,
            risk_aversion=1.0,
            long_only=True,
        )

        for ticker, weight in weights.items():
            if weight > 0:
                all_weights.append({
                    "date": date,
                    "ticker": ticker,
                    "weight": weight,
                })

    weights_df = pl.DataFrame(all_weights)
    print(f"\n   Generated weights: {weights_df.shape}")

    # Show sample
    if not weights_df.is_empty():
        sample_date = weights_df["date"].unique().sort()[0]
        sample = weights_df.filter(pl.col("date") == sample_date).sort("weight", descending=True).head(10)
        print(f"\n   Sample weights ({sample_date}):")
        for row in sample.iter_rows(named=True):
            print(f"      {row['ticker']:>6}: {row['weight']*100:>6.2f}%")

    # ========================================
    # 3. Run backtest
    # ========================================
    print("\n🚀 Running backtest...")

    constraints = Constraints(
        min_position_value=1.0,      # Min $1 per position
        min_trade_value=1.0,         # Min $1 per trade
        max_position_pct=0.10,       # Max 10% per position
        max_turnover=1.0,            # Allow full rebalance daily
        allow_short=False,           # Long only
    )

    bt = Backtest(
        initial_capital=100_000,
        constraints=constraints,
        slippage_bps=5.0,
        commission_bps=10.0,
    )

    result = bt.run(prices=prices, weights=weights_df)

    # ========================================
    # 4. Show results
    # ========================================
    print(result.summary())

    print("📈 Equity Curve (last 10 days):")
    print(result.equity_curve.tail(10))

    print(f"\n📊 Total trades: {len(result.trades)}")
    if not result.trades.is_empty():
        print(f"   Trading costs: ${result.trades['cost'].sum():,.2f}")

    print("\n🏆 Final positions (top 10):")
    if not result.positions.is_empty():
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
