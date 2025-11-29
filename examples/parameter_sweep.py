#!/usr/bin/env python3
"""Parameter sweep example - grid search optimization with parallel execution.

Demonstrates: Systematic testing of parameter combinations to find optimal settings.
Runtime: ~5-15 seconds (depending on CPU cores)
"""
from pathlib import Path

import numpy as np

from simple_futures_backtester.backtest.sweep import ParameterSweep
from simple_futures_backtester.config import BacktestConfig, SweepConfig
from simple_futures_backtester.data.loader import load_csv
from simple_futures_backtester.strategy.base import register_strategy
from simple_futures_backtester.strategy.examples import MomentumStrategy

# Register strategy - required for sweep to find it by name
register_strategy("momentum", MomentumStrategy)


def main() -> None:
    # Step 1: Load OHLCV data
    data_path = Path(__file__).parent / "sample_data" / "es_1min_sample.csv"
    print(f"Loading data from: {data_path}")
    df = load_csv(data_path)
    print(f"Loaded {len(df):,} bars")

    # Step 2: Extract numpy arrays
    open_arr = df["open"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    volume_arr = df["volume"].values.astype(np.int64)

    # Step 3: Define parameter grid for optimization
    # Each parameter gets a list of values to test - all combinations are evaluated
    sweep_config = SweepConfig(
        strategy="momentum",
        parameters={
            "rsi_period": [10, 14, 20],     # 3 values
            "fast_ema": [5, 9, 12],         # 3 values
            "slow_ema": [21, 30, 50],       # 3 values = 27 total combinations
        },
        backtest_overrides={"fees": 0.0001, "slippage": 0.0001},
    )

    n_combos = 1
    for vals in sweep_config.parameters.values():
        n_combos *= len(vals)
    print(f"\nTesting {n_combos} parameter combinations...")

    # Step 4: Run parallel sweep using all CPU cores
    # n_jobs=-1 uses all available cores for maximum throughput
    sweeper = ParameterSweep(n_jobs=-1)
    backtest_config = BacktestConfig(initial_capital=100_000.0, freq="1min")

    # Optional progress callback for tracking
    def on_progress(current: int, total: int) -> None:
        if current % 5 == 0 or current == total:
            print(f"  Progress: {current}/{total} ({100*current/total:.0f}%)")

    result = sweeper.run(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
        sweep_config=sweep_config,
        base_backtest_config=backtest_config,
        progress_callback=on_progress,
    )

    # Step 5: Display top 5 parameter combinations by Sharpe ratio
    print(f"\n{'='*70}")
    print("Top 5 Parameter Combinations by Sharpe Ratio")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'RSI':<8}{'Fast':<8}{'Slow':<8}{'Sharpe':<10}{'Return':<10}{'Trades':<8}")
    print(f"{'-'*70}")

    for rank, (params, bt_result) in enumerate(result.all_results[:5], start=1):
        print(
            f"{rank:<6}"
            f"{params['rsi_period']:<8}"
            f"{params['fast_ema']:<8}"
            f"{params['slow_ema']:<8}"
            f"{bt_result.sharpe_ratio:<10.3f}"
            f"{bt_result.total_return:<10.2%}"
            f"{bt_result.n_trades:<8}"
        )

    print(f"{'-'*70}")
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Best Sharpe Ratio: {result.best_sharpe:.3f}")
    print(f"Total combinations tested: {len(result.all_results)}")


if __name__ == "__main__":
    main()
