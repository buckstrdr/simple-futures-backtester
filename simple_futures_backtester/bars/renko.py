"""Renko bar generation with JIT compilation.

Renko bars are price bars formed by fixed-size bricks. A new brick forms when
the price moves by brick_size or more. Reversals require 2x brick_size movement.

Supports both fixed brick size and ATR-based dynamic sizing.

Performance: Targets 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars.renko import generate_renko_bars_series
    >>> bars = generate_renko_bars_series(
    ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
    ...     brick_size=10.0
    ... )
    >>> print(f"Generated {len(bars)} Renko bars")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.bars import BarSeries, register_bar_type
from simple_futures_backtester.utils.jit_utils import (
    get_njit_decorator,
    validate_ohlcv_arrays,
)

if TYPE_CHECKING:
    pass


# Get JIT decorator with project defaults
_jit = get_njit_decorator(cache=True, parallel=False)


@_jit  # pragma: no cover
def _generate_renko_bars_nb(
    high: NDArray[np.float64],  # noqa: ARG001
    low: NDArray[np.float64],  # noqa: ARG001
    close: NDArray[np.float64],
    brick_size: float,
    atr_length: int,
    atr_values: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],  # open
    NDArray[np.float64],  # high
    NDArray[np.float64],  # low
    NDArray[np.float64],  # close
    NDArray[np.int64],  # bar_indices
]:
    """JIT-compiled Renko bar generator core algorithm.

    Implements the Renko bar formation algorithm with support for fixed brick
    size and ATR-based dynamic sizing. Uses the standard 2-brick reversal rule
    where trend reversals require price to move 2x brick distance.

    Args:
        high: High prices array (currently unused, included for future extensions).
        low: Low prices array (currently unused, included for future extensions).
        close: Close prices array used for brick formation decisions.
        brick_size: Fixed brick size. Ignored when atr_values is provided and
            atr_length > 0.
        atr_length: ATR lookback period. When > 0 and atr_values provided,
            uses dynamic brick sizing from atr_values[i].
        atr_values: Pre-computed ATR values for dynamic sizing. When provided
            with atr_length > 0, brick size at index i equals atr_values[i].

    Returns:
        Tuple of (open, high, low, close, bar_indices) numpy arrays.
        - open, high, low, close: Renko bar OHLC values (float64)
        - bar_indices: Source row index where each bar completed (int64)

    Note:
        This is the low-level JIT implementation. Use generate_renko_bars_series()
        for the high-level API with validation and BarSeries output.
    """
    n = len(close)

    # Pre-allocate output arrays
    # Worst case: a single tick can create multiple bars if price jumps significantly.
    # Estimate max bars as: n * average_bricks_per_tick, but cap reasonably.
    # For safety, use max(n, price_range / brick_size) as upper bound.
    if n > 0:
        price_range = np.abs(close.max() - close.min())
        # Minimum brick_size to avoid division issues
        effective_brick = max(brick_size, 1e-10)
        range_based_max = int(price_range / effective_brick) + n
        max_bars = max(n, range_based_max)
    else:
        max_bars = n
    renko_open = np.empty(max_bars, dtype=np.float64)
    renko_high = np.empty(max_bars, dtype=np.float64)
    renko_low = np.empty(max_bars, dtype=np.float64)
    renko_close = np.empty(max_bars, dtype=np.float64)
    bar_indices = np.empty(max_bars, dtype=np.int64)

    # Initialize state
    bar_count = 0
    current_open = close[0]
    current_direction = 0  # 0 = neutral, 1 = up, -1 = down

    for i in range(1, n):
        # Get brick size (fixed or ATR-based)
        if len(atr_values) > 0 and atr_length > 0 and i >= atr_length:
            brick = atr_values[i]
        else:
            brick = brick_size

        price = close[i]

        # Check for new brick
        if current_direction >= 0:  # Looking for up or first brick
            if price >= current_open + brick:
                # Up brick - create as many bricks as price allows
                while price >= current_open + brick:
                    renko_open[bar_count] = current_open
                    renko_close[bar_count] = current_open + brick
                    renko_high[bar_count] = current_open + brick
                    renko_low[bar_count] = current_open
                    bar_indices[bar_count] = i
                    current_open = current_open + brick
                    current_direction = 1
                    bar_count += 1

            elif price <= current_open - brick and current_direction == 0:
                # First brick is down (only when neutral)
                while price <= current_open - brick:
                    renko_open[bar_count] = current_open
                    renko_close[bar_count] = current_open - brick
                    renko_high[bar_count] = current_open
                    renko_low[bar_count] = current_open - brick
                    bar_indices[bar_count] = i
                    current_open = current_open - brick
                    current_direction = -1
                    bar_count += 1

        if current_direction == 1:  # Was going up, check for reversal or continuation
            if price <= current_open - 2 * brick:
                # Reversal down (needs 2 bricks distance)
                current_open = current_open - brick  # Step back one brick
                while price <= current_open - brick:
                    renko_open[bar_count] = current_open
                    renko_close[bar_count] = current_open - brick
                    renko_high[bar_count] = current_open
                    renko_low[bar_count] = current_open - brick
                    bar_indices[bar_count] = i
                    current_open = current_open - brick
                    current_direction = -1
                    bar_count += 1

        elif current_direction == -1:  # Was going down
            if price <= current_open - brick:
                # Continue down
                while price <= current_open - brick:
                    renko_open[bar_count] = current_open
                    renko_close[bar_count] = current_open - brick
                    renko_high[bar_count] = current_open
                    renko_low[bar_count] = current_open - brick
                    bar_indices[bar_count] = i
                    current_open = current_open - brick
                    bar_count += 1

            elif price >= current_open + 2 * brick:
                # Reversal up (needs 2 bricks distance)
                current_open = current_open + brick  # Step back one brick
                while price >= current_open + brick:
                    renko_open[bar_count] = current_open
                    renko_close[bar_count] = current_open + brick
                    renko_high[bar_count] = current_open + brick
                    renko_low[bar_count] = current_open
                    bar_indices[bar_count] = i
                    current_direction = 1
                    current_open = current_open + brick
                    bar_count += 1

    # Trim arrays to actual size
    return (
        renko_open[:bar_count],
        renko_high[:bar_count],
        renko_low[:bar_count],
        renko_close[:bar_count],
        bar_indices[:bar_count],
    )


def generate_renko_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    brick_size: float | None = None,
    atr_length: int = 0,
    atr_values: NDArray[np.float64] | None = None,
) -> BarSeries:
    """Generate Renko bars from OHLCV data.

    High-level wrapper that validates inputs, calls the JIT-compiled core
    algorithm, aggregates volume per bar, and returns a BarSeries.

    Renko bars are price-based bars that filter out noise by only creating
    new bars when price moves by a minimum amount (brick_size). This results
    in bars of uniform size, making trend identification easier.

    Args:
        open_arr: Opening prices (included in BarSeries but not used for
            Renko brick calculation).
        high_arr: High prices (passed to JIT function for future extensions).
        low_arr: Low prices (passed to JIT function for future extensions).
        close_arr: Closing prices used for brick formation decisions.
        volume_arr: Volume data to be aggregated per Renko bar.
        brick_size: Fixed brick size. Required when atr_values is None.
            Each new brick forms when price moves >= brick_size.
        atr_length: ATR lookback period for dynamic sizing. When > 0 and
            atr_values provided, uses atr_values[i] as brick size at each bar.
        atr_values: Pre-computed ATR values for dynamic brick sizing.
            When provided with atr_length > 0, enables dynamic sizing.

    Returns:
        BarSeries containing Renko bars with:
        - type: "renko"
        - parameters: {"brick_size": float, "atr_length": int}
        - OHLCV arrays for the generated bars
        - index_map: Source row where each bar completed

    Raises:
        ValueError: If brick_size is None/invalid and atr_values not provided.
        ValueError: If brick_size <= 0.
        ValueError: If arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> import numpy as np
        >>> close = np.array([100, 105, 115, 120, 110, 95, 100])
        >>> bars = generate_renko_bars_series(
        ...     open_arr=close, high_arr=close, low_arr=close,
        ...     close_arr=close, volume_arr=np.ones(7, dtype=np.int64) * 100,
        ...     brick_size=10.0
        ... )
        >>> print(f"Generated {len(bars)} Renko bars")
    """
    # Validate inputs
    open_arr, high_arr, low_arr, close_arr, volume_arr = validate_ohlcv_arrays(
        open_arr, high_arr, low_arr, close_arr, volume_arr
    )

    n = len(close_arr)

    # Handle edge cases
    if n < 2:
        return BarSeries(
            type="renko",
            parameters={"brick_size": brick_size or 0.0, "atr_length": atr_length},
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.int64),
            index_map=np.array([], dtype=np.int64),
        )

    # Validate brick_size or atr_values
    if atr_values is None or len(atr_values) == 0:
        if brick_size is None:
            raise ValueError("brick_size is required when atr_values is not provided")
        if brick_size <= 0:
            raise ValueError(f"brick_size must be positive, got {brick_size}")
        # Create empty atr_values for JIT function
        atr_values_jit = np.empty(0, dtype=np.float64)
    else:
        atr_values_jit = np.ascontiguousarray(atr_values, dtype=np.float64)
        if brick_size is None:
            brick_size = 1.0  # Placeholder, won't be used with valid atr_values

    # Call JIT-compiled core algorithm
    renko_open, renko_high, renko_low, renko_close, bar_indices = _generate_renko_bars_nb(
        high_arr,
        low_arr,
        close_arr,
        brick_size,
        atr_length,
        atr_values_jit,
    )

    # Aggregate volume for each Renko bar
    num_bars = len(bar_indices)
    renko_volume = np.zeros(num_bars, dtype=np.int64)

    if num_bars > 0:
        # First bar aggregates from start to its completion index
        renko_volume[0] = np.sum(volume_arr[: bar_indices[0] + 1])

        # Subsequent bars aggregate from previous bar's index + 1 to current index
        for bar_idx in range(1, num_bars):
            start_idx = bar_indices[bar_idx - 1] + 1
            end_idx = bar_indices[bar_idx] + 1
            renko_volume[bar_idx] = np.sum(volume_arr[start_idx:end_idx])

    # Determine actual brick_size for parameters
    actual_brick_size = brick_size if brick_size is not None else 0.0

    return BarSeries(
        type="renko",
        parameters={"brick_size": actual_brick_size, "atr_length": atr_length},
        open=renko_open,
        high=renko_high,
        low=renko_low,
        close=renko_close,
        volume=renko_volume,
        index_map=bar_indices,
    )


# Auto-register on module import
register_bar_type("renko", generate_renko_bars_series)


__all__: list[str] = [
    "generate_renko_bars_series",
]
