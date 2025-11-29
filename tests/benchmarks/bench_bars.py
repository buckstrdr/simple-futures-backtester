"""Performance benchmarks for all bar generators.

Consolidated benchmark suite for bar generation throughput testing.
All bar generators must achieve 1M+ rows/sec throughput.

Benchmarks:
- Renko bars: Fixed brick size (missing from individual files)
- Range bars: Price range threshold
- Tick bars: Fixed tick count aggregation
- Volume bars: Volume threshold aggregation
- Dollar bars: Dollar volume threshold
- Tick Imbalance bars: Tick direction imbalance
- Volume Imbalance bars: Volume-weighted imbalance

Run with: pytest tests/benchmarks/bench_bars.py -v
Run only bar benchmarks: pytest tests/benchmarks/ -m "benchmark and bars"
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from simple_futures_backtester.bars.renko import generate_renko_bars_series
from simple_futures_backtester.bars.range_bars import generate_range_bars_series
from simple_futures_backtester.bars.tick_bars import generate_tick_bars_series
from simple_futures_backtester.bars.volume_bars import generate_volume_bars_series
from simple_futures_backtester.bars.dollar_bars import generate_dollar_bars_series
from simple_futures_backtester.bars.imbalance_bars import (
    generate_tick_imbalance_bars_series,
    generate_volume_imbalance_bars_series,
)


def _generate_benchmark_data(n: int = 1_000_000) -> tuple:
    """Generate deterministic benchmark OHLCV data.

    Uses seed=42 for reproducibility across runs. Creates realistic
    price data with random walk close, realistic high/low spreads,
    and random volume.

    Args:
        n: Number of rows to generate (default 1M).

    Returns:
        Tuple of (open_arr, high_arr, low_arr, close_arr, volume_arr).
    """
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_arr = close.copy()
    volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)
    return open_arr, high, low, close, volume


# =============================================================================
# RENKO BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_renko_bars_1m_rows_performance() -> None:
    """Benchmark Renko bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses fixed brick_size=5.0 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT (first run triggers compilation)
    _ = generate_renko_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        brick_size=5.0,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_renko_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        brick_size=5.0,
    )
    elapsed = time.perf_counter() - start

    # Calculate throughput
    throughput = n / elapsed
    print(f"\nRenko bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    # Verify meets performance target
    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# RANGE BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_range_bars_1m_rows_performance() -> None:
    """Benchmark Range bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses range_size=5.0 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_range_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        range_size=5.0,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_range_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        range_size=5.0,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nRange bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# TICK BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_tick_bars_1m_rows_performance() -> None:
    """Benchmark Tick bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses tick_threshold=100 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_tick_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        tick_threshold=100,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_tick_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        tick_threshold=100,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nTick bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# VOLUME BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_volume_bars_1m_rows_performance() -> None:
    """Benchmark Volume bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses volume_threshold=100000 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_volume_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        volume_threshold=100000,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_volume_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        volume_threshold=100000,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nVolume bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# DOLLAR BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_dollar_bars_1m_rows_performance() -> None:
    """Benchmark Dollar bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses dollar_threshold=10_000_000 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_dollar_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        dollar_threshold=10_000_000.0,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_dollar_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        dollar_threshold=10_000_000.0,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nDollar bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# TICK IMBALANCE BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_tick_imbalance_bars_1m_rows_performance() -> None:
    """Benchmark Tick Imbalance bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses imbalance_threshold=100 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_tick_imbalance_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        imbalance_threshold=100,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_tick_imbalance_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        imbalance_threshold=100,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nTick Imbalance bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# VOLUME IMBALANCE BARS BENCHMARK
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.bars
def test_volume_imbalance_bars_1m_rows_performance() -> None:
    """Benchmark Volume Imbalance bar generation with 1M rows.

    Target: >= 1,000,000 rows/sec throughput.
    Uses imbalance_threshold=100000 for consistent benchmarking.
    """
    n = 1_000_000
    open_arr, high, low, close, volume = _generate_benchmark_data(n)

    # Warm up JIT
    _ = generate_volume_imbalance_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        imbalance_threshold=100000,
    )

    # Measure performance
    start = time.perf_counter()
    result = generate_volume_imbalance_bars_series(
        open_arr=open_arr,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        volume_arr=volume,
        imbalance_threshold=100000,
    )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\nVolume Imbalance bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Bar Generation Performance Benchmarks")
    print("=" * 60)
    print("\nRunning all bar benchmarks with 1M rows...")
    print("Target: >= 1,000,000 rows/sec throughput\n")

    test_renko_bars_1m_rows_performance()
    test_range_bars_1m_rows_performance()
    test_tick_bars_1m_rows_performance()
    test_volume_bars_1m_rows_performance()
    test_dollar_bars_1m_rows_performance()
    test_tick_imbalance_bars_1m_rows_performance()
    test_volume_imbalance_bars_1m_rows_performance()

    print("\n" + "=" * 60)
    print("All bar benchmarks completed successfully!")
    print("=" * 60)
