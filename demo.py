#!/usr/bin/env python3
"""
Harbinger Demo — MVO with 5% active risk target.

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
    load_benchmark_weights,
)
from harbinger.optimize import (
    build_factor_covariance,
    mean_variance_with_risk_target,
    signal_to_expected_returns,
)
from harbinger.backtest import Backtest, Constraints


def main():
    print("🔮 Harbinger Demo — 5% Active Risk Target\n")
    print("=" * 60)

    # ========================================
    # 1. Load ALL data upfront
    # ========================================
    print("\n📊 Loading data from Bear Lake...")

    client = get_bear_lake_client()

    start_date = "2023-01-01"
    end_date = "2026-01-31"

    prices = load_prices(client, start_date, end_date)
    print(f"   Prices: {prices.shape}")

    signals_raw = load_signals(client, start_date, end_date, signal_name="reversal")
    print(f"   Signals (raw): {signals_raw.shape}")
    
    # LAG SIGNALS BY 1 DAY to avoid look-ahead bias
    # Signal from day T-1 is used to trade on day T
    signals = signals_raw.with_columns([
        pl.col("date").shift(-1).over("ticker").alias("trade_date")
    ]).drop("date").rename({"trade_date": "date"}).drop_nulls()
    print(f"   Signals (lagged): {signals.shape}")

    factor_loadings = load_factor_loadings(client, start_date, end_date)
    print(f"   Factor loadings: {factor_loadings.shape}")

    factor_covariances = load_factor_covariances(client, start_date, end_date)
    print(f"   Factor covariances: {factor_covariances.shape}")

    idio_vol = load_idio_vol(client, start_date, end_date)
    print(f"   Idio vol: {idio_vol.shape}")

    benchmark_weights = load_benchmark_weights(client, start_date, end_date)
    print(f"   Benchmark weights: {benchmark_weights.shape}")

    # Get trading dates
    signal_dates = set(signals["date"].unique().to_list())
    factor_dates = set(factor_loadings["date"].unique().to_list())
    idio_dates = set(idio_vol["date"].unique().to_list())
    bench_dates = set(benchmark_weights["date"].unique().to_list())
    trading_dates = sorted(signal_dates & factor_dates & idio_dates & bench_dates)
    print(f"   Trading dates: {len(trading_dates)}")

    # ========================================
    # 2. Generate daily weights with 5% risk target
    # ========================================
    print("\n⚖️  Running daily MVO with 5% active risk target...")

    all_weights = []
    all_metrics = []

    for i, date in enumerate(trading_dates):
        if i % 20 == 0:
            print(f"   Processing date {i+1}/{len(trading_dates)}: {date}")

        # Build covariance matrix
        cov_matrix, tickers = build_factor_covariance(
            factor_loadings, factor_covariances, idio_vol, date=date
        )

        if len(tickers) == 0:
            continue

        # Get expected returns
        expected_returns = signal_to_expected_returns(signals, date, scale=1.0)
        expected_returns = {k: v for k, v in expected_returns.items() if k in tickers}

        if not expected_returns:
            continue

        # Get benchmark weights for this date
        bench_df = benchmark_weights.filter(pl.col("date") == date)
        bench_dict = {
            row["ticker"]: row["weight"]
            for row in bench_df.iter_rows(named=True)
            if row["ticker"] in tickers
        }

        # Run MVO with 5% active risk target
        weights, final_lambda, achieved_risk = mean_variance_with_risk_target(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            tickers=tickers,
            benchmark_weights=bench_dict,
            target_active_risk=0.05,
            long_only=True,
        )

        for ticker, weight in weights.items():
            if weight > 0:
                all_weights.append({
                    "date": date,
                    "ticker": ticker,
                    "weight": weight,
                })

        all_metrics.append({
            "date": date,
            "lambda": final_lambda,
            "active_risk": achieved_risk,
        })

    weights_df = pl.DataFrame(all_weights)
    metrics_df = pl.DataFrame(all_metrics)

    print(f"\n   Generated weights: {weights_df.shape}")

    # Show risk targeting results
    avg_risk = metrics_df["active_risk"].mean()
    print(f"   Average active risk: {avg_risk*100:.2f}% (target: 5.00%)")
    print(f"   Risk range: [{metrics_df['active_risk'].min()*100:.2f}%, {metrics_df['active_risk'].max()*100:.2f}%]")

    # ========================================
    # 3. Run backtest
    # ========================================
    print("\n🚀 Running backtest...")

    constraints = Constraints(
        min_position_value=1.0,
        min_trade_value=1.0,
        max_position_pct=0.10,
        max_turnover=1.0,
        allow_short=False,
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
