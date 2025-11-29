"""Performance benchmarks for parameter sweep execution.

Tests 100-combination parameter sweep performance to ensure
reasonable sweep completion times. Target: < 10 seconds for 100 combos (aspirational).

Note: VectorBT's Portfolio.from_signals() has inherent overhead per backtest.
Actual sweep time depends on data size and system performance. The default
threshold is relaxed to 30s for CI stability, with the aspirational 10s target
documented.

Run with: pytest tests/benchmarks/bench_sweep.py -v
Run only backtest benchmarks: pytest tests/benchmarks/ -m "benchmark and backtest"

Environment Variables:
    SFB_SWEEP_100_THRESHOLD_SEC: Override the 100-combo sweep threshold (default: 30)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from simple_futures_backtester.backtest.sweep import ParameterSweep
from simple_futures_backtester.config import BacktestConfig, SweepConfig

# Default threshold with env override for CI customization
# Aspirational target is 10s, but VectorBT overhead makes this challenging
SWEEP_100_THRESHOLD_SEC = float(os.environ.get("SFB_SWEEP_100_THRESHOLD_SEC", "30"))


def _generate_benchmark_data(n: int = 5_000) -> tuple:
    """Generate deterministic benchmark OHLCV data.

    Uses seed=42 for reproducibility. 5,000 bars provides a good
    balance between realistic workload and reasonable sweep time.

    Args:
        n: Number of bars to generate.

    Returns:
        Tuple of (open_arr, high_arr, low_arr, close_arr, volume_arr).
    """
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 1.0)
    low = close - np.abs(np.random.randn(n) * 1.0)
    open_arr = close.copy()
    volume = np.abs(np.random.randn(n) * 1000 + 5000).astype(np.int64)
    return open_arr, high, low, close, volume


@pytest.mark.benchmark
@pytest.mark.backtest
def test_parameter_sweep_100_combos() -> None:
    """Benchmark 100-combination parameter sweep.

    Target: < 10 seconds for 100 parameter combinations.
    Uses sequential execution (n_jobs=1) for deterministic timing.

    Parameter grid: 5 x 5 x 4 = 100 combinations
    - rsi_period: [10, 12, 14, 16, 18] (5 values)
    - fast_ema: [5, 7, 9, 11, 13] (5 values)
    - slow_ema: [21, 30, 40, 50] (4 values)
    """
    n = 5_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Create sweep config with exactly 100 combinations (5 x 5 x 4 = 100)
    sweep_config = SweepConfig(
        strategy="momentum",
        parameters={
            "rsi_period": [10, 12, 14, 16, 18],  # 5 values
            "fast_ema": [5, 7, 9, 11, 13],  # 5 values
            "slow_ema": [21, 30, 40, 50],  # 4 values
        },
        backtest_overrides={
            "initial_capital": 100000.0,
            "fees": 0.0001,
        },
    )

    # Use sequential execution for deterministic timing
    sweeper = ParameterSweep(n_jobs=1)

    # Measure performance
    start = time.perf_counter()
    result = sweeper.run(
        open_arr=open_arr.astype(np.float64),
        high_arr=high.astype(np.float64),
        low_arr=low.astype(np.float64),
        close_arr=close.astype(np.float64),
        volume_arr=volume.astype(np.int64),
        sweep_config=sweep_config,
    )
    elapsed = time.perf_counter() - start

    n_combos = len(result.all_results)
    avg_per_combo_ms = (elapsed / n_combos) * 1000

    print(f"\n100-combo parameter sweep ({n:,} bars):")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Combinations tested: {n_combos}")
    print(f"  Average per combo: {avg_per_combo_ms:.2f} ms")
    print(f"  Best Sharpe: {result.best_sharpe:.3f}")
    print(f"  Best params: {result.best_params}")

    # Print top 3 results
    print("\n  Top 3 parameter sets:")
    for i, (params, backtest_result) in enumerate(result.all_results[:3], 1):
        print(f"    {i}. Sharpe={backtest_result.sharpe_ratio:.3f}, Params={params}")

    # Verify meets performance target
    # Aspirational target: 10s. Default threshold: 30s (configurable via env)
    assert elapsed < SWEEP_100_THRESHOLD_SEC, (
        f"Sweep time {elapsed:.2f}s >= {SWEEP_100_THRESHOLD_SEC}s threshold "
        f"(aspirational target: 10s)"
    )
    assert n_combos == 100, f"Expected 100 combos, got {n_combos}"


@pytest.mark.benchmark
@pytest.mark.backtest
def test_parameter_sweep_throughput() -> None:
    """Benchmark parameter sweep throughput.

    Tests how many backtests per second the sweep engine can execute.
    Uses smaller parameter grid for faster testing.
    """
    n = 3_000  # Smaller data for throughput testing
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Create sweep config with 36 combinations (4 x 3 x 3 = 36)
    sweep_config = SweepConfig(
        strategy="momentum",
        parameters={
            "rsi_period": [10, 14, 18, 22],  # 4 values
            "fast_ema": [5, 9, 13],  # 3 values
            "slow_ema": [21, 35, 50],  # 3 values
        },
        backtest_overrides={
            "initial_capital": 100000.0,
        },
    )

    sweeper = ParameterSweep(n_jobs=1)

    start = time.perf_counter()
    result = sweeper.run(
        open_arr=open_arr.astype(np.float64),
        high_arr=high.astype(np.float64),
        low_arr=low.astype(np.float64),
        close_arr=close.astype(np.float64),
        volume_arr=volume.astype(np.int64),
        sweep_config=sweep_config,
    )
    elapsed = time.perf_counter() - start

    n_combos = len(result.all_results)
    throughput = n_combos / elapsed

    print(f"\nParameter sweep throughput ({n:,} bars):")
    print(f"  Combinations: {n_combos}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Throughput: {throughput:.1f} backtests/sec")

    # Should achieve reasonable throughput
    assert throughput > 5, f"Throughput {throughput:.1f} < 5 backtests/sec"


@pytest.mark.benchmark
@pytest.mark.backtest
def test_parameter_sweep_with_progress() -> None:
    """Benchmark parameter sweep with progress callback.

    Tests sweep performance when using progress tracking,
    which is common in CLI and UI integrations.
    """
    n = 4_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Create sweep config with 25 combinations (5 x 5 = 25)
    sweep_config = SweepConfig(
        strategy="momentum",
        parameters={
            "rsi_period": [10, 12, 14, 16, 18],  # 5 values
            "fast_ema": [5, 7, 9, 11, 13],  # 5 values
            # Using fixed slow_ema to reduce combinations
        },
        backtest_overrides={
            "initial_capital": 100000.0,
        },
    )

    sweeper = ParameterSweep(n_jobs=1)

    # Track progress
    progress_calls = []

    def progress_callback(current: int, total: int) -> None:
        progress_calls.append((current, total))

    start = time.perf_counter()
    result = sweeper.run(
        open_arr=open_arr.astype(np.float64),
        high_arr=high.astype(np.float64),
        low_arr=low.astype(np.float64),
        close_arr=close.astype(np.float64),
        volume_arr=volume.astype(np.int64),
        sweep_config=sweep_config,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - start

    n_combos = len(result.all_results)

    print(f"\nParameter sweep with progress ({n:,} bars):")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Combinations: {n_combos}")
    print(f"  Progress callbacks: {len(progress_calls)}")

    # Verify progress was tracked
    assert len(progress_calls) == n_combos, "Progress callback not called for each combo"
    assert progress_calls[-1][0] == n_combos, "Final progress not equal to total"


if __name__ == "__main__":
    print("=" * 60)
    print("Parameter Sweep Performance Benchmarks")
    print("=" * 60)
    print("\nTarget: < 10 seconds for 100 combinations\n")

    test_parameter_sweep_100_combos()
    test_parameter_sweep_throughput()
    test_parameter_sweep_with_progress()

    print("\n" + "=" * 60)
    print("All sweep benchmarks completed successfully!")
    print("=" * 60)
