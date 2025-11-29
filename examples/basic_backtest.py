#!/usr/bin/env python3
"""Basic backtest example - load data, run momentum strategy, print results.

Demonstrates: Minimal workflow from raw data to backtest metrics.
Runtime: ~2 seconds
"""
from pathlib import Path

import numpy as np

from simple_futures_backtester.backtest.engine import BacktestEngine
from simple_futures_backtester.config import BacktestConfig, StrategyConfig
from simple_futures_backtester.data.loader import load_csv
from simple_futures_backtester.strategy.base import get_strategy, register_strategy
from simple_futures_backtester.strategy.examples import MomentumStrategy

# Register the momentum strategy so it's available in the registry
register_strategy("momentum", MomentumStrategy)


def main() -> None:
    # Step 1: Load OHLCV data from CSV
    # The loader validates schema and normalizes dtypes automatically
    data_path = Path(__file__).parent / "sample_data" / "es_1min_sample.csv"
    print(f"Loading data from: {data_path}")
    df = load_csv(data_path)
    print(f"Loaded {len(df):,} bars of E-mini S&P 500 data")

    # Step 2: Extract numpy arrays for strategy and engine
    open_arr = df["open"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    volume_arr = df["volume"].values.astype(np.int64)

    # Step 3: Configure and instantiate strategy
    # RSI + EMA momentum strategy: long when RSI>50 and fast EMA > slow EMA
    strategy_config = StrategyConfig(
        name="momentum",
        parameters={"rsi_period": 14, "fast_ema": 9, "slow_ema": 21},
    )
    StrategyClass = get_strategy("momentum")
    strategy = StrategyClass(strategy_config)
    print(f"Strategy: {strategy_config.name} with params {strategy_config.parameters}")

    # Step 4: Generate trading signals (-1=short, 0=flat, 1=long)
    signals = strategy.generate_signals(open_arr, high_arr, low_arr, close_arr, volume_arr)
    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    print(f"Generated signals: {n_long:,} long bars, {n_short:,} short bars")

    # Step 5: Run backtest with realistic futures trading costs
    backtest_config = BacktestConfig(
        initial_capital=100_000.0,
        fees=0.0001,      # 0.01% commission per trade
        slippage=0.0001,  # 0.01% slippage per trade
        size=1,           # 1 contract per signal
        size_type="fixed",
        freq="1min",      # 1-minute bar frequency
    )
    engine = BacktestEngine()
    result = engine.run(close_arr, signals, backtest_config)

    # Step 6: Display results
    print(f"\n{'='*60}")
    print(f"Backtest Results: Momentum Strategy on ES Futures")
    print(f"{'='*60}")
    print(f"Total Return:     {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"Win Rate:         {result.win_rate:>10.2%}")
    print(f"Profit Factor:    {result.profit_factor:>10.2f}")
    print(f"Number of Trades: {result.n_trades:>10}")
    print(f"Avg Trade PnL:    {result.avg_trade:>10.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
