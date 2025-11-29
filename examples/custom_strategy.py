#!/usr/bin/env python3
"""Custom strategy example - implement your own strategy extending BaseStrategy.

Demonstrates: How to create a custom trading strategy with the framework.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from simple_futures_backtester.backtest.engine import BacktestEngine
from simple_futures_backtester.config import BacktestConfig, StrategyConfig
from simple_futures_backtester.data.loader import load_csv
from simple_futures_backtester.strategy.base import BaseStrategy, register_strategy


class SMACrossoverStrategy(BaseStrategy):
    """SMA crossover: long when fast > slow, short when fast < slow."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.fast_period: int = self.config.parameters.get("fast_period", 10)
        self.slow_period: int = self.config.parameters.get("slow_period", 30)

    def generate_signals(
        self, open_arr: NDArray[np.float64], high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64], close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        signals = np.zeros(len(close_arr), dtype=np.int32)
        close_series = pd.Series(close_arr)
        fast_sma = close_series.rolling(window=self.fast_period).mean().values
        slow_sma = close_series.rolling(window=self.slow_period).mean().values
        signals[fast_sma > slow_sma] = 1   # Long in uptrend
        signals[fast_sma < slow_sma] = -1  # Short in downtrend
        signals[:max(self.fast_period, self.slow_period)] = 0  # Warmup period
        return signals


def main() -> None:
    # Step 1: Register our custom strategy
    register_strategy("sma_crossover", SMACrossoverStrategy)
    print("Registered custom strategy: sma_crossover")

    # Step 2: Load data
    data_path = Path(__file__).parent / "sample_data" / "es_1min_sample.csv"
    df = load_csv(data_path)
    print(f"Loaded {len(df):,} bars from {data_path.name}")

    # Step 3: Extract arrays
    open_arr = df["open"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    volume_arr = df["volume"].values.astype(np.int64)

    # Step 4: Create and run custom strategy
    strategy_config = StrategyConfig(
        name="sma_crossover",
        parameters={"fast_period": 10, "slow_period": 30},
    )
    strategy = SMACrossoverStrategy(strategy_config)
    signals = strategy.generate_signals(open_arr, high_arr, low_arr, close_arr, volume_arr)
    print(f"Signals: {np.sum(signals == 1):,} long, {np.sum(signals == -1):,} short")

    # Step 5: Backtest
    config = BacktestConfig(initial_capital=100_000.0, fees=0.0001, freq="1min")
    result = BacktestEngine().run(close_arr, signals, config)

    # Step 6: Display results
    print(f"\n{'='*60}")
    print("Custom SMA Crossover Strategy Results")
    print(f"{'='*60}")
    print(f"Parameters: fast={strategy.fast_period}, slow={strategy.slow_period}")
    print(f"{'-'*60}")
    print(f"Total Return:     {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"Win Rate:         {result.win_rate:>10.2%}")
    print(f"Number of Trades: {result.n_trades:>10}")
    print(f"{'='*60}")
    print("\nTo customize: modify fast_period/slow_period or add new logic!")


if __name__ == "__main__":
    main()
