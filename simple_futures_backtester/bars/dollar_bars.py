"""Dollar bar generation with JIT compilation.

Dollar bars aggregate source bars until cumulative dollar volume reaches a threshold.
Dollar volume is calculated as close[i] * volume[i] for each source bar. Each bar
closes when the accumulated dollar volume meets or exceeds dollar_threshold.

This bar type creates bars based on monetary trading activity rather than time,
price movement, or raw volume, which can help normalize bars across periods
of varying price levels and activity.

Performance: Targets 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars.dollar_bars import generate_dollar_bars_series
    >>> bars = generate_dollar_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     dollar_threshold=1000000.0
    ... )
    >>> print(f"Generated {len(bars)} Dollar bars")
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


@_jit
def _generate_dollar_bars_nb(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    dollar_threshold: np.float64,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # bar_indices
    NDArray[np.int64],  # start_indices (for volume aggregation)
]:
    """JIT-compiled Dollar bar generator core algorithm.

    Accumulates dollar volume (close * volume) from source bars until
    cumulative_dollars >= dollar_threshold, then closes the bar and resets.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume for each source bar.
        dollar_threshold: Dollar volume threshold for closing bars.

    Returns:
        Tuple of (open, high, low, close, bar_indices, start_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use generate_dollar_bars_series()
        for the high-level API with validation and BarSeries output.

    """
    n = len(close_arr)

    # Pre-allocate (worst case: every source bar creates a dollar bar)
    max_bars = n
    bar_open = np.empty(max_bars, dtype=np.float64)
    bar_high = np.empty(max_bars, dtype=np.float64)
    bar_low = np.empty(max_bars, dtype=np.float64)
    bar_close = np.empty(max_bars, dtype=np.float64)
    bar_indices = np.empty(max_bars, dtype=np.int64)
    start_indices = np.empty(max_bars, dtype=np.int64)

    # Initialize state with first bar's values
    bar_count = 0
    current_open = open_arr[0]
    current_high = high_arr[0]
    current_low = low_arr[0]
    current_start_idx = np.int64(0)

    # Calculate dollar volume for first bar
    cumulative_dollars = close_arr[0] * np.float64(volume_arr[0])

    for i in range(1, n):
        # Update cumulative high/low
        if high_arr[i] > current_high:
            current_high = high_arr[i]
        if low_arr[i] < current_low:
            current_low = low_arr[i]

        # Accumulate dollar volume: close[i] * volume[i]
        dollar_volume = close_arr[i] * np.float64(volume_arr[i])
        cumulative_dollars += dollar_volume

        # Check if dollar threshold reached
        if cumulative_dollars >= dollar_threshold:
            # Close bar
            bar_open[bar_count] = current_open
            bar_high[bar_count] = current_high
            bar_low[bar_count] = current_low
            bar_close[bar_count] = close_arr[i]
            bar_indices[bar_count] = i
            start_indices[bar_count] = current_start_idx
            bar_count += 1

            # Reset for next bar - start from close price of current bar
            # This ensures continuity and no gaps in coverage
            current_open = close_arr[i]
            current_high = high_arr[i]
            current_low = low_arr[i]
            current_start_idx = np.int64(i + 1)
            cumulative_dollars = np.float64(0.0)

    # Trim arrays to actual size
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_indices[:bar_count],
        start_indices[:bar_count],
    )


def generate_dollar_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    dollar_threshold: float,
) -> BarSeries:
    """Generate Dollar bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, aggregates raw volume per bar, and returns a BarSeries.

    Dollar bars accumulate dollar volume (close * volume) from source bars,
    closing when the cumulative dollar volume meets or exceeds dollar_threshold.
    This creates bars based on monetary trading activity, which normalizes
    for both price level and trading activity.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data for each source bar.
        dollar_threshold: Dollar volume threshold for closing bars. A new bar
            is created when cumulative (close * volume) meets or exceeds this.

    Returns:
        BarSeries containing Dollar bars with:
        - type: "dollar"
        - parameters: {"dollar_threshold": float}
        - OHLCV arrays for the generated bars (volume is aggregated raw volume)
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If dollar_threshold <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Price data with volume
        >>> close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        >>> volume = np.array([100, 150, 200, 250, 300, 350, 400], dtype=np.int64)
        >>> # Dollar volume per bar: 10000, 15150, 20400, 25750, 31200, 36750, 42400
        >>> bars = generate_dollar_bars_series(
        ...     open_arr=close, high_arr=close + 1, low_arr=close - 1,
        ...     close_arr=close, volume_arr=volume,
        ...     dollar_threshold=50000.0
        ... )
        >>> print(f"Generated {len(bars)} Dollar bars")

    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="dollar",
            parameters={"dollar_threshold": dollar_threshold},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if dollar_threshold <= 0:
        raise ValueError(f"dollar_threshold must be positive, got {dollar_threshold}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_indices, start_indices = (
        _generate_dollar_bars_nb(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            np.float64(dollar_threshold),
        )
    )

    # Aggregate raw volume for each Dollar bar
    num_bars = len(bar_indices)
    aggregated_volume = np.zeros(num_bars, dtype=np.int64)

    if num_bars > 0:
        for bar_idx in range(num_bars):
            start_idx = start_indices[bar_idx]
            end_idx = bar_indices[bar_idx] + 1
            aggregated_volume[bar_idx] = np.sum(volume_arr[start_idx:end_idx])

    return BarSeries(
        type="dollar",
        parameters={"dollar_threshold": dollar_threshold},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=aggregated_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("dollar", generate_dollar_bars_series)


__all__: list[str] = [
    "generate_dollar_bars_series",
]
