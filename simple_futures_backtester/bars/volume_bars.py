"""Volume bar generation with JIT compilation.

Volume bars aggregate source bars until cumulative volume reaches a threshold.
Each bar closes when the accumulated volume from source bars meets or exceeds
the volume_threshold, then resets for the next bar.

This bar type creates bars based on trading activity rather than time or price,
which can help normalize bars across periods of high and low activity.

Performance: Targets 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars.volume_bars import generate_volume_bars_series
    >>> bars = generate_volume_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     volume_threshold=10000
    ... )
    >>> print(f"Generated {len(bars)} Volume bars")
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
def _generate_volume_bars_nb(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    volume_threshold: np.int64,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # cumulative_volume per bar
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Volume bar generator core algorithm.

    Accumulates volume from source bars until cumulative_volume >= volume_threshold,
    then closes the bar and resets for the next bar.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume for each source bar.
        volume_threshold: Volume threshold for closing bars.

    Returns:
        Tuple of (open, high, low, close, cumulative_volume, bar_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use generate_volume_bars_series()
        for the high-level API with validation and BarSeries output.

    """
    n = len(close_arr)

    # Pre-allocate (worst case: every source bar creates a volume bar)
    max_bars = n
    bar_open = np.empty(max_bars, dtype=np.float64)
    bar_high = np.empty(max_bars, dtype=np.float64)
    bar_low = np.empty(max_bars, dtype=np.float64)
    bar_close = np.empty(max_bars, dtype=np.float64)
    bar_volume = np.empty(max_bars, dtype=np.int64)
    bar_indices = np.empty(max_bars, dtype=np.int64)

    # Initialize state with first bar's values
    bar_count = 0
    current_open = open_arr[0]
    current_high = high_arr[0]
    current_low = low_arr[0]
    cumulative_volume = volume_arr[0]

    for i in range(1, n):
        # Update cumulative high/low
        if high_arr[i] > current_high:
            current_high = high_arr[i]
        if low_arr[i] < current_low:
            current_low = low_arr[i]

        # Accumulate volume
        cumulative_volume += volume_arr[i]

        # Check if volume threshold reached
        if cumulative_volume >= volume_threshold:
            # Close bar
            bar_open[bar_count] = current_open
            bar_high[bar_count] = current_high
            bar_low[bar_count] = current_low
            bar_close[bar_count] = close_arr[i]
            bar_volume[bar_count] = cumulative_volume
            bar_indices[bar_count] = i
            bar_count += 1

            # Reset for next bar - start from close price of current bar
            # This ensures continuity and no gaps in coverage
            current_open = close_arr[i]
            current_high = high_arr[i]
            current_low = low_arr[i]
            cumulative_volume = np.int64(0)

    # Trim arrays to actual size
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_volume[:bar_count],
        bar_indices[:bar_count],
    )


def generate_volume_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    volume_threshold: int,
) -> BarSeries:
    """Generate Volume bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, and returns a BarSeries.

    Volume bars accumulate volume from source bars, closing when the cumulative
    volume meets or exceeds volume_threshold. This creates bars based on trading
    activity rather than time, which can help normalize volatility across periods
    of high and low activity.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data for each source bar.
        volume_threshold: Volume threshold for closing bars. A new bar is created
            when the cumulative volume meets or exceeds this value.

    Returns:
        BarSeries containing Volume bars with:
        - type: "volume"
        - parameters: {"volume_threshold": int}
        - OHLCV arrays for the generated bars (volume is cumulative per bar)
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If volume_threshold <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Price data with varying volume
        >>> close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        >>> volume = np.array([500, 600, 700, 800, 900, 1000, 1100], dtype=np.int64)
        >>> bars = generate_volume_bars_series(
        ...     open_arr=close, high_arr=close + 1, low_arr=close - 1,
        ...     close_arr=close, volume_arr=volume,
        ...     volume_threshold=2000
        ... )
        >>> print(f"Generated {len(bars)} Volume bars")

    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="volume",
            parameters={"volume_threshold": volume_threshold},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if volume_threshold <= 0:
        raise ValueError(f"volume_threshold must be positive, got {volume_threshold}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_indices = (
        _generate_volume_bars_nb(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            np.int64(volume_threshold),
        )
    )

    return BarSeries(
        type="volume",
        parameters={"volume_threshold": volume_threshold},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=bar_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("volume", generate_volume_bars_series)


__all__: list[str] = [
    "generate_volume_bars_series",
]
