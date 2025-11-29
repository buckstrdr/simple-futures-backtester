"""Tick and Volume Imbalance bar generation with JIT compilation.

Imbalance bars aggregate source bars based on the cumulative imbalance between
upticks and downticks. They capture market microstructure by measuring the
directional flow of trades.

Two types are provided:

1. **Tick Imbalance Bars**: Accumulate signed tick direction (+1 for uptick,
   -1 for downtick, 0 for unchanged). Bar closes when |cumulative_imbalance|
   >= imbalance_threshold.

2. **Volume Imbalance Bars**: Accumulate volume-weighted tick direction
   (tick_direction * volume). Bar closes when |cumulative_imbalance|
   >= imbalance_threshold.

These bar types are useful for detecting periods of strong buying or selling
pressure and can reveal market structure not visible in time-based bars.

Performance: Both generators target 1M+ rows/sec throughput via Numba JIT.

Usage:
    >>> from simple_futures_backtester.bars.imbalance_bars import (
    ...     generate_tick_imbalance_bars_series,
    ...     generate_volume_imbalance_bars_series,
    ... )
    >>> # Tick imbalance bars
    >>> tick_bars = generate_tick_imbalance_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     imbalance_threshold=50
    ... )
    >>> print(f"Generated {len(tick_bars)} Tick Imbalance bars")
    >>>
    >>> # Volume imbalance bars
    >>> vol_bars = generate_volume_imbalance_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     imbalance_threshold=100000
    ... )
    >>> print(f"Generated {len(vol_bars)} Volume Imbalance bars")
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
def _generate_tick_imbalance_bars_nb(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    imbalance_threshold: np.int64,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # cumulative_volume per bar
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Tick Imbalance bar generator core algorithm.

    Tracks the signed tick direction (+1 for uptick, -1 for downtick, 0 for
    unchanged) and accumulates imbalance. Bar closes when the absolute value
    of cumulative imbalance reaches or exceeds the threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume for each source bar.
        imbalance_threshold: Absolute imbalance threshold for closing bars.

    Returns:
        Tuple of (open, high, low, close, cumulative_volume, bar_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use
        generate_tick_imbalance_bars_series() for the high-level API with
        validation and BarSeries output.
    """
    n = len(close_arr)

    # Pre-allocate (worst case: every source bar creates an imbalance bar)
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
    cumulative_imbalance = np.int64(0)

    for i in range(1, n):
        # Determine tick direction by comparing close to previous close
        if close_arr[i] > close_arr[i - 1]:
            tick_direction = np.int64(1)
        elif close_arr[i] < close_arr[i - 1]:
            tick_direction = np.int64(-1)
        else:
            tick_direction = np.int64(0)

        # Accumulate imbalance
        cumulative_imbalance += tick_direction

        # Track high/low
        if high_arr[i] > current_high:
            current_high = high_arr[i]
        if low_arr[i] < current_low:
            current_low = low_arr[i]

        # Track volume
        cumulative_volume += volume_arr[i]

        # Check if imbalance threshold reached (absolute value)
        if cumulative_imbalance >= imbalance_threshold or cumulative_imbalance <= -imbalance_threshold:
            # Close bar
            bar_open[bar_count] = current_open
            bar_high[bar_count] = current_high
            bar_low[bar_count] = current_low
            bar_close[bar_count] = close_arr[i]
            bar_volume[bar_count] = cumulative_volume
            bar_indices[bar_count] = i
            bar_count += 1

            # Reset for next bar - start from close price of current bar
            current_open = close_arr[i]
            current_high = high_arr[i]
            current_low = low_arr[i]
            cumulative_volume = np.int64(0)
            cumulative_imbalance = np.int64(0)

    # Trim arrays to actual size
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_volume[:bar_count],
        bar_indices[:bar_count],
    )


def generate_tick_imbalance_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    imbalance_threshold: int,
) -> BarSeries:
    """Generate Tick Imbalance bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, and returns a BarSeries.

    Tick Imbalance bars track market microstructure by accumulating signed tick
    directions. Each tick is classified as:
    - +1 (uptick): close > previous close
    - -1 (downtick): close < previous close
    - 0 (no change): close == previous close

    A new bar is created when |cumulative_imbalance| >= imbalance_threshold,
    indicating a period of sustained buying (+) or selling (-) pressure.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data for each source bar.
        imbalance_threshold: Absolute imbalance threshold for closing bars.
            A new bar is created when the absolute cumulative tick imbalance
            meets or exceeds this value.

    Returns:
        BarSeries containing Tick Imbalance bars with:
        - type: "tick_imbalance"
        - parameters: {"imbalance_threshold": int}
        - OHLCV arrays for the generated bars (volume is cumulative per bar)
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If imbalance_threshold <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Simulated price data with trending movements
        >>> close = np.array([100.0, 101.0, 102.0, 103.0, 102.5, 103.5, 104.5])
        >>> volume = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600], dtype=np.int64)
        >>> bars = generate_tick_imbalance_bars_series(
        ...     open_arr=close - 0.5, high_arr=close + 0.5, low_arr=close - 1.0,
        ...     close_arr=close, volume_arr=volume,
        ...     imbalance_threshold=3
        ... )
        >>> print(f"Generated {len(bars)} Tick Imbalance bars")
    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="tick_imbalance",
            parameters={"imbalance_threshold": imbalance_threshold},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if imbalance_threshold <= 0:
        raise ValueError(f"imbalance_threshold must be positive, got {imbalance_threshold}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_indices = (
        _generate_tick_imbalance_bars_nb(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            np.int64(imbalance_threshold),
        )
    )

    return BarSeries(
        type="tick_imbalance",
        parameters={"imbalance_threshold": imbalance_threshold},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=bar_volume,
        index_map=bar_indices,
    )


@_jit
def _generate_volume_imbalance_bars_nb(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    imbalance_threshold: np.int64,
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # cumulative_volume per bar
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Volume Imbalance bar generator core algorithm.

    Tracks the volume-weighted tick direction (tick_direction * volume) and
    accumulates imbalance. Bar closes when the absolute value of cumulative
    imbalance reaches or exceeds the threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume for each source bar.
        imbalance_threshold: Absolute volume imbalance threshold for closing bars.

    Returns:
        Tuple of (open, high, low, close, cumulative_volume, bar_indices) arrays.

    Note:
        This is the low-level JIT implementation. Use
        generate_volume_imbalance_bars_series() for the high-level API with
        validation and BarSeries output.
    """
    n = len(close_arr)

    # Pre-allocate (worst case: every source bar creates an imbalance bar)
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
    cumulative_imbalance = np.int64(0)

    for i in range(1, n):
        # Determine tick direction by comparing close to previous close
        if close_arr[i] > close_arr[i - 1]:
            tick_direction = np.int64(1)
        elif close_arr[i] < close_arr[i - 1]:
            tick_direction = np.int64(-1)
        else:
            tick_direction = np.int64(0)

        # Accumulate volume-weighted imbalance
        cumulative_imbalance += tick_direction * volume_arr[i]

        # Track high/low
        if high_arr[i] > current_high:
            current_high = high_arr[i]
        if low_arr[i] < current_low:
            current_low = low_arr[i]

        # Track volume (unsigned for bar output)
        cumulative_volume += volume_arr[i]

        # Check if imbalance threshold reached (absolute value)
        if cumulative_imbalance >= imbalance_threshold or cumulative_imbalance <= -imbalance_threshold:
            # Close bar
            bar_open[bar_count] = current_open
            bar_high[bar_count] = current_high
            bar_low[bar_count] = current_low
            bar_close[bar_count] = close_arr[i]
            bar_volume[bar_count] = cumulative_volume
            bar_indices[bar_count] = i
            bar_count += 1

            # Reset for next bar - start from close price of current bar
            current_open = close_arr[i]
            current_high = high_arr[i]
            current_low = low_arr[i]
            cumulative_volume = np.int64(0)
            cumulative_imbalance = np.int64(0)

    # Trim arrays to actual size
    return (
        bar_open[:bar_count],
        bar_high[:bar_count],
        bar_low[:bar_count],
        bar_close[:bar_count],
        bar_volume[:bar_count],
        bar_indices[:bar_count],
    )


def generate_volume_imbalance_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    imbalance_threshold: int,
) -> BarSeries:
    """Generate Volume Imbalance bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, and returns a BarSeries.

    Volume Imbalance bars extend tick imbalance by weighting each tick by its
    volume. Each tick's contribution is:
    - +volume (uptick): close > previous close
    - -volume (downtick): close < previous close
    - 0 (no change): close == previous close

    A new bar is created when |cumulative_imbalance| >= imbalance_threshold,
    indicating a period of sustained volume-weighted buying (+) or selling (-)
    pressure.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data for each source bar.
        imbalance_threshold: Absolute volume imbalance threshold for closing bars.
            A new bar is created when the absolute cumulative volume-weighted
            imbalance meets or exceeds this value.

    Returns:
        BarSeries containing Volume Imbalance bars with:
        - type: "volume_imbalance"
        - parameters: {"imbalance_threshold": int}
        - OHLCV arrays for the generated bars (volume is cumulative per bar)
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If imbalance_threshold <= 0 or arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> # Simulated price data with high-volume moves
        >>> close = np.array([100.0, 101.0, 102.0, 101.5, 102.5, 103.5, 104.0])
        >>> volume = np.array([10000, 15000, 20000, 5000, 25000, 30000, 10000], dtype=np.int64)
        >>> bars = generate_volume_imbalance_bars_series(
        ...     open_arr=close - 0.5, high_arr=close + 0.5, low_arr=close - 1.0,
        ...     close_arr=close, volume_arr=volume,
        ...     imbalance_threshold=50000
        ... )
        >>> print(f"Generated {len(bars)} Volume Imbalance bars")
    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr,
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="volume_imbalance",
            parameters={"imbalance_threshold": imbalance_threshold},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    if imbalance_threshold <= 0:
        raise ValueError(f"imbalance_threshold must be positive, got {imbalance_threshold}")

    # Call JIT-compiled core algorithm
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_indices = (
        _generate_volume_imbalance_bars_nb(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            np.int64(imbalance_threshold),
        )
    )

    return BarSeries(
        type="volume_imbalance",
        parameters={"imbalance_threshold": imbalance_threshold},
        open=bar_open,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=bar_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("tick_imbalance", generate_tick_imbalance_bars_series)
register_bar_type("volume_imbalance", generate_volume_imbalance_bars_series)


__all__: list[str] = [
    "generate_tick_imbalance_bars_series",
    "generate_volume_imbalance_bars_series",
]
