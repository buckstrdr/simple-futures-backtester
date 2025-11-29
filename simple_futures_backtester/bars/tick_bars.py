"""Tick bar generation with JIT compilation.

Tick bars aggregate N source bars into each output bar. This is the simplest
alternative bar type, useful as a baseline for comparing other bar types and
for downsampling data by a fixed factor.

Unlike time-based bars, tick bars ensure each output bar contains exactly the
same number of source bars (except possibly the final bar), which can help
normalize volatility across bars.

Performance: Targets 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars.tick_bars import generate_tick_bars_series
    >>> bars = generate_tick_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     tick_threshold=10
    ... )
    >>> print(f"Generated {len(bars)} Tick bars")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.bars import BarSeries, register_bar_type
from simple_futures_backtester.utils.jit_utils import (
    get_njit_decorator,
    validate_ohlcv_arrays,
)

# Get JIT decorator with project defaults
_jit = get_njit_decorator(cache=True, parallel=False)


@_jit  # pragma: no cover
def _generate_tick_bars_nb(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    tick_threshold: int,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Tick bar generator core algorithm.

    Aggregates every tick_threshold source bars into one output bar.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        tick_threshold: Number of source bars per output bar.

    Returns:
        Tuple of (open, high, low, close, bar_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use generate_tick_bars_series()
        for the high-level API with validation and BarSeries output.

    """
    n = len(close_arr)

    # Calculate exact number of bars needed (ceiling division)
    max_bars = (n + tick_threshold - 1) // tick_threshold
    bar_open = np.empty(max_bars, dtype=np.float64)
    bar_high = np.empty(max_bars, dtype=np.float64)
    bar_low = np.empty(max_bars, dtype=np.float64)
    bar_close = np.empty(max_bars, dtype=np.float64)
    bar_indices = np.empty(max_bars, dtype=np.int64)

    bar_count = 0

    # Process source bars in groups of tick_threshold
    start_idx = 0
    while start_idx < n:
        end_idx = start_idx + tick_threshold
        end_idx = min(end_idx, n)

        # Aggregate OHLC for this tick bar
        bar_open[bar_count] = open_arr[start_idx]
        bar_close[bar_count] = close_arr[end_idx - 1]
        bar_indices[bar_count] = end_idx - 1

        # Find high and low across the group
        group_high = high_arr[start_idx]
        group_low = low_arr[start_idx]
        for i in range(start_idx + 1, end_idx):
            if high_arr[i] > group_high:
                group_high = high_arr[i]
            if low_arr[i] < group_low:
                group_low = low_arr[i]

        bar_high[bar_count] = group_high
        bar_low[bar_count] = group_low

        bar_count += 1
        start_idx = end_idx

    # Arrays should already be exact size, but trim for safety
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_indices[:bar_count],
    )


def generate_tick_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    tick_threshold: int,
) -> BarSeries:
    """Generate Tick bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, aggregates volume per bar, and returns a BarSeries.

    Tick bars aggregate every tick_threshold source bars into one output bar.
    This is the simplest alternative bar type, effectively downsampling by
    a fixed factor while preserving OHLCV aggregation.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data to be aggregated per Tick bar.
        tick_threshold: Number of source bars per output bar. Must be positive.

    Returns:
        BarSeries containing Tick bars with:
        - type: "tick"
        - parameters: {"tick_threshold": int}
        - OHLCV arrays for the generated bars
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If tick_threshold <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # 15 source bars aggregated into 5-bar ticks
        >>> close = np.arange(100.0, 115.0)  # 15 bars
        >>> bars = generate_tick_bars_series(
        ...     open_arr=close, high_arr=close + 1, low_arr=close - 1,
        ...     close_arr=close, volume_arr=np.ones(15, dtype=np.int64) * 100,
        ...     tick_threshold=5
        ... )
        >>> print(f"Generated {len(bars)} Tick bars")  # Should be 3

    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n == 0:
        return BarSeries(
            type="tick",
            parameters={"tick_threshold": tick_threshold},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if tick_threshold <= 0:
        raise ValueError(f"tick_threshold must be positive, got {tick_threshold}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_indices = _generate_tick_bars_nb(
        open_arr, high_arr, low_arr, close_arr, tick_threshold,
    )

    # Aggregate volume for each Tick bar
    num_bars = len(bar_indices)
    aggregated_volume = np.zeros(num_bars, dtype=np.int64)

    if num_bars > 0:
        # First bar aggregates from start to its completion index
        aggregated_volume[0] = np.sum(volume_arr[: bar_indices[0] + 1])

        # Subsequent bars aggregate from previous bar's index + 1 to current index
        for bar_idx in range(1, num_bars):
            start_idx = bar_indices[bar_idx - 1] + 1
            end_idx = bar_indices[bar_idx] + 1
            aggregated_volume[bar_idx] = np.sum(volume_arr[start_idx:end_idx])

    return BarSeries(
        type="tick",
        parameters={"tick_threshold": tick_threshold},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=aggregated_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("tick", generate_tick_bars_series)


__all__: list[str] = [
    "generate_tick_bars_series",
]
