"""Range bar generation with JIT compilation.

Range bars form when the high-low range exceeds a threshold. Each bar tracks
cumulative high/low across source bars until range_size is exceeded, then
closes and resets.

This bar type filters intrabar noise by only creating new bars when price
volatility reaches the specified threshold, making it useful for trading
strategies that need to ignore minor fluctuations.

Performance: Targets 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars.range_bars import generate_range_bars_series
    >>> bars = generate_range_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     range_size=10.0
    ... )
    >>> print(f"Generated {len(bars)} Range bars")
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
def _generate_range_bars_nb(  # pragma: no cover
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    range_size: float,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Range bar generator core algorithm.

    Tracks cumulative high/low across source bars. Closes bar when
    current_high - current_low >= range_size.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        range_size: Range threshold for closing bars.

    Returns:
        Tuple of (open, high, low, close, bar_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use generate_range_bars_series()
        for the high-level API with validation and BarSeries output.

    """
    n = len(close_arr)

    # Pre-allocate (worst case: every source bar creates a range bar)
    max_bars = n
    bar_open = np.empty(max_bars, dtype=np.float64)
    bar_high = np.empty(max_bars, dtype=np.float64)
    bar_low = np.empty(max_bars, dtype=np.float64)
    bar_close = np.empty(max_bars, dtype=np.float64)
    bar_indices = np.empty(max_bars, dtype=np.int64)

    # Initialize state with first bar's values
    bar_count = 0
    current_open = open_arr[0]
    current_high = high_arr[0]
    current_low = low_arr[0]

    for i in range(1, n):
        # Update cumulative high/low
        if high_arr[i] > current_high:
            current_high = high_arr[i]
        if low_arr[i] < current_low:
            current_low = low_arr[i]

        # Check if range exceeded
        if current_high - current_low >= range_size:
            # Close bar
            bar_open[bar_count] = current_open
            bar_high[bar_count] = current_high
            bar_low[bar_count] = current_low
            bar_close[bar_count] = close_arr[i]
            bar_indices[bar_count] = i
            bar_count += 1

            # Reset for next bar - start from close price of current bar
            # This ensures continuity and no gaps in coverage
            current_open = close_arr[i]
            current_high = high_arr[i]
            current_low = low_arr[i]

    # Trim arrays to actual size
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_indices[:bar_count],
    )


def generate_range_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    range_size: float,
) -> BarSeries:
    """Generate Range bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, aggregates volume per bar, and returns a BarSeries.

    Range bars track cumulative high/low across source bars, closing when
    the range (high - low) exceeds range_size. This filters intrabar noise
    and creates bars based on actual price movement rather than time.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data to be aggregated per Range bar.
        range_size: Range threshold (in price units). A new bar is created
            when the cumulative high minus low exceeds this value.

    Returns:
        BarSeries containing Range bars with:
        - type: "range"
        - parameters: {"range_size": float}
        - OHLCV arrays for the generated bars
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If range_size <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Price data with 10-point range moves
        >>> close = np.array([100.0, 105.0, 110.0, 108.0, 115.0, 120.0, 118.0])
        >>> high = close + 2
        >>> low = close - 2
        >>> bars = generate_range_bars_series(
        ...     open_arr=close, high_arr=high, low_arr=low,
        ...     close_arr=close, volume_arr=np.ones(7, dtype=np.int64) * 100,
        ...     range_size=10.0
        ... )
        >>> print(f"Generated {len(bars)} Range bars")

    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="range",
            parameters={"range_size": range_size},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if range_size <= 0:
        raise ValueError(f"range_size must be positive, got {range_size}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_indices = _generate_range_bars_nb(
        open_arr, high_arr, low_arr, close_arr, range_size,
    )

    # Aggregate volume for each Range bar
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
        type="range",
        parameters={"range_size": range_size},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=aggregated_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("range", generate_range_bars_series)


__all__: list[str] = [
    "generate_range_bars_series",
]
