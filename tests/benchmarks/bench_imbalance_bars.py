"""Performance benchmarks for Tick and Volume Imbalance bar generation.

Benchmarks target 1M+ rows/sec throughput for JIT-compiled functions.

Run with: pytest tests/benchmarks/bench_imbalance_bars.py -v
"""

from __future__ import annotations

import time

import numpy as np

from simple_futures_backtester.bars.imbalance_bars import (
    generate_tick_imbalance_bars_series,
    generate_volume_imbalance_bars_series,
)


def test_tick_imbalance_bars_1m_rows_performance() -> None:
    """Measure throughput for 1M rows of tick imbalance bar generation."""
    # Generate 1M rows of realistic price data
    n = 1_000_000
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_arr = close
    volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

    # Warm up JIT (first run triggers compilation)
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

    # Calculate throughput
    throughput = n / elapsed
    print(f"\nTick Imbalance bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    # Verify meets performance target
    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


def test_volume_imbalance_bars_1m_rows_performance() -> None:
    """Measure throughput for 1M rows of volume imbalance bar generation."""
    # Generate 1M rows of realistic price and volume data
    n = 1_000_000
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_arr = close
    volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

    # Warm up JIT (first run triggers compilation)
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

    # Calculate throughput
    throughput = n / elapsed
    print(f"\nVolume Imbalance bars throughput: {throughput:,.0f} rows/sec")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(f"Generated {len(result)} bars from {n:,} source bars")

    # Verify meets performance target
    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec target"
    assert len(result) > 0, "Should generate at least some bars"


def test_tick_imbalance_bars_100k_rows() -> None:
    """Benchmark tick imbalance with 100K rows (faster test)."""
    n = 100_000
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_arr = close
    volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

    # Warm up
    _ = generate_tick_imbalance_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        imbalance_threshold=100,
    )

    # Measure
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
    print(f"\n100K Tick Imbalance benchmark - Throughput: {throughput:,.0f} rows/sec")
    assert len(result) > 0


def test_volume_imbalance_bars_100k_rows() -> None:
    """Benchmark volume imbalance with 100K rows (faster test)."""
    n = 100_000
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_arr = close
    volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

    # Warm up
    _ = generate_volume_imbalance_bars_series(
        open_arr=open_arr[:1000],
        high_arr=high[:1000],
        low_arr=low[:1000],
        close_arr=close[:1000],
        volume_arr=volume[:1000],
        imbalance_threshold=100000,
    )

    # Measure
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
    print(f"\n100K Volume Imbalance benchmark - Throughput: {throughput:,.0f} rows/sec")
    assert len(result) > 0


if __name__ == "__main__":
    test_tick_imbalance_bars_100k_rows()
    test_volume_imbalance_bars_100k_rows()
    test_tick_imbalance_bars_1m_rows_performance()
    test_volume_imbalance_bars_1m_rows_performance()
