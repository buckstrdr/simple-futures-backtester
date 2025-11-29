#!/usr/bin/env python3
"""Renko bar strategy example - generate alternative bars for noise filtering.

Demonstrates: Using Renko bars to compress noisy price data into cleaner trends.
Runtime: ~3 seconds
"""
from pathlib import Path

import numpy as np

from simple_futures_backtester.backtest.engine import BacktestEngine
from simple_futures_backtester.bars.renko import generate_renko_bars_series
from simple_futures_backtester.config import BacktestConfig, StrategyConfig
from simple_futures_backtester.data.loader import load_csv
from simple_futures_backtester.strategy.base import register_strategy
from simple_futures_backtester.strategy.examples import MomentumStrategy

# Register strategy for use
register_strategy("momentum", MomentumStrategy)


def main() -> None:
    # Step 1: Load source 1-minute OHLCV data
    data_path = Path(__file__).parent / "sample_data" / "es_1min_sample.csv"
    print(f"Loading source data from: {data_path}")
    df = load_csv(data_path)
    source_bars = len(df)
    print(f"Loaded {source_bars:,} 1-minute bars")

    # Step 2: Extract numpy arrays from source data
    open_arr = df["open"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    volume_arr = df["volume"].values.astype(np.int64)

    # Step 3: Generate Renko bars with fixed brick size
    # Renko bars filter market noise by only forming new bars when price moves
    # by a minimum amount (brick_size). This creates bars of uniform size.
    brick_size = 2.0  # Each brick = 2 points of price movement
    print(f"\nGenerating Renko bars with brick_size={brick_size}...")

    renko_bars = generate_renko_bars_series(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
        brick_size=brick_size,
    )

    renko_count = len(renko_bars)
    compression = source_bars / renko_count if renko_count > 0 else float("inf")
    print(f"Generated {renko_count:,} Renko bars ({compression:.1f}x compression)")

    # Step 4: Run strategy on Renko bars (not source data!)
    # This gives cleaner signals by filtering out intrabar noise
    strategy_config = StrategyConfig(
        name="momentum",
        parameters={"rsi_period": 14, "fast_ema": 9, "slow_ema": 21},
    )
    strategy = MomentumStrategy(strategy_config)

    # Generate signals using Renko OHLCV arrays
    signals = strategy.generate_signals(
        renko_bars.open,
        renko_bars.high,
        renko_bars.low,
        renko_bars.close,
        renko_bars.volume,
    )
    print(f"Generated {np.sum(signals == 1):,} long and {np.sum(signals == -1):,} short signals")

    # Step 5: Run backtest on Renko data
    backtest_config = BacktestConfig(
        initial_capital=100_000.0,
        fees=0.0001,
        slippage=0.0001,
        size=1,
        freq="1D",  # Renko bars don't have fixed time frequency
    )
    engine = BacktestEngine()
    result = engine.run(renko_bars.close, signals, backtest_config)

    # Step 6: Display results
    print(f"\n{'='*60}")
    print(f"Renko Strategy Results (brick_size={brick_size})")
    print(f"{'='*60}")
    print(f"Source bars:      {source_bars:>10,}")
    print(f"Renko bars:       {renko_count:>10,}")
    print(f"Compression:      {compression:>10.1f}x")
    print(f"{'-'*60}")
    print(f"Total Return:     {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"Win Rate:         {result.win_rate:>10.2%}")
    print(f"Number of Trades: {result.n_trades:>10}")
    print(f"{'='*60}")
    print("\nNote: Renko bars filter noise, often improving signal quality")
    print("by removing small price fluctuations that create false signals.")


if __name__ == "__main__":
    main()
