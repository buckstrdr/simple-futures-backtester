"""JIT-compiled trailing stop generators for VectorBT integration.

Provides two types of trailing stops that VectorBT does not natively support:

1. **Delayed Trailing Stop** (delayed_trailing_stop_nb):
   A trailing stop that only activates after price moves favorably by a
   specified activation threshold. Once activated, it trails the peak price
   by a configurable trail percentage.

2. **ATR Trailing Stop** (atr_trailing_stop_nb):
   A trailing stop that uses Average True Range (ATR) for dynamic stop
   distance. The stop trails immediately from entry with distance = ATR * multiplier.

Both functions are JIT-compiled for 1M+ rows/sec throughput.

Usage:
    >>> from simple_futures_backtester.extensions.trailing_stops import (
    ...     delayed_trailing_stop_nb,
    ...     atr_trailing_stop_nb,
    ...     generate_trailing_exits,
    ... )
    >>>
    >>> # Delayed trailing stop for a long position
    >>> exit_idx, exit_price, peak = delayed_trailing_stop_nb(
    ...     close, entry_price=100.0, entry_idx=10,
    ...     trail_percent=0.02, activation_percent=0.01, direction=1
    ... )
    >>>
    >>> # ATR trailing stop for a short position
    >>> exit_idx, exit_price = atr_trailing_stop_nb(
    ...     high, low, close, atr,
    ...     entry_idx=10, atr_mult=2.0, direction=-1
    ... )
    >>>
    >>> # Generate exit signal array for VectorBT
    >>> exits = generate_trailing_exits(
    ...     entries, close, entry_prices,
    ...     trail_percent=0.02, activation_percent=0.01, direction=1
    ... )
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.utils.jit_utils import get_njit_decorator

# Get JIT decorator with project defaults
_jit = get_njit_decorator(cache=True, parallel=False)


@_jit  # pragma: no cover
def delayed_trailing_stop_nb(
    close: NDArray[np.float64],
    entry_price: float,
    entry_idx: int,
    trail_percent: float,
    activation_percent: float,
    direction: int,
) -> tuple[int, float, float]:
    """Trailing stop that only activates after price moves in favor.

    Implements a two-phase trailing stop:
    1. **Activation Phase**: Wait for price to move favorably by activation_percent
       before the trailing stop becomes active.
    2. **Trailing Phase**: Once activated, track peak price and exit when price
       retraces by trail_percent from the peak.

    This is useful for letting winners run before protecting profits, avoiding
    premature exits from normal market noise.

    Args:
        close: Close prices array (float64, C-contiguous).
        entry_price: Entry price for the position.
        entry_idx: Bar index where entry occurred (0-based).
        trail_percent: Trail distance as decimal (e.g., 0.02 = 2%). Once activated,
            stop is placed at peak * (1 - trail_percent) for longs or
            trough * (1 + trail_percent) for shorts.
        activation_percent: Required favorable move before trail activates
            (e.g., 0.01 = 1%). For longs, price must reach entry * (1 + activation_percent).
            For shorts, price must reach entry * (1 - activation_percent).
        direction: Position direction. 1 = long (profit when price rises),
            -1 = short (profit when price falls).

    Returns:
        Tuple of (exit_idx, exit_price, peak_price):
        - exit_idx: Bar index where stop was triggered, or -1 if no exit.
        - exit_price: Price at exit, or 0.0 if no exit.
        - peak_price: Best price reached (highest for long, lowest for short).

    Example:
        >>> import numpy as np
        >>> close = np.array([100.0, 101.0, 103.0, 102.0, 100.5, 99.0])
        >>> # Long position: activate at +1%, trail at 2%
        >>> exit_idx, exit_price, peak = delayed_trailing_stop_nb(
        ...     close, entry_price=100.0, entry_idx=0,
        ...     trail_percent=0.02, activation_percent=0.01, direction=1
        ... )
        >>> # Price hits 103 (activates at 101), trails from 103
        >>> # Stop at 103 * 0.98 = 100.94, triggers at 100.5? No, at 99.0
    """
    n = len(close)

    # Edge case: no bars after entry
    if entry_idx >= n - 1:
        return (-1, 0.0, entry_price)

    activation_price = entry_price * (1 + direction * activation_percent)
    activated = False
    peak = entry_price

    for i in range(entry_idx + 1, n):
        price = close[i]

        # Check activation
        if not activated and (
            (direction == 1 and price >= activation_price)
            or (direction == -1 and price <= activation_price)
        ):
            activated = True
            peak = price

        # Update peak and check stop
        if activated:
            if direction == 1:
                peak = max(peak, price)
                stop_price = peak * (1 - trail_percent)
                if price <= stop_price:
                    return (i, price, peak)
            else:
                peak = min(peak, price)
                stop_price = peak * (1 + trail_percent)
                if price >= stop_price:
                    return (i, price, peak)

    return (-1, 0.0, peak)  # No exit


@_jit  # pragma: no cover
def atr_trailing_stop_nb(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr: NDArray[np.float64],
    entry_idx: int,
    atr_mult: float,
    direction: int,
) -> tuple[int, float]:
    """ATR-based trailing stop with dynamic stop distance.

    Implements a trailing stop where the distance from peak to stop is based on
    the Average True Range (ATR) multiplied by a configurable factor. This adapts
    the stop distance to current market volatility.

    Unlike delayed_trailing_stop_nb, this activates immediately from entry -
    there is no activation threshold. The stop trails from the first bar.

    Args:
        high: High prices array (float64, C-contiguous). Used for tracking
            peak prices on long positions.
        low: Low prices array (float64, C-contiguous). Used for tracking
            trough prices on short positions.
        close: Close prices array (float64, C-contiguous). Used for exit
            price determination.
        atr: Pre-computed ATR array (float64, C-contiguous). Must be computed
            externally, e.g., via VectorBT: `vbt.ATR.run(high, low, close, 14).atr.values`.
        entry_idx: Bar index where entry occurred (0-based).
        atr_mult: ATR multiplier for stop distance (e.g., 2.0 = 2x ATR).
            Stop is placed at peak - (atr_mult * ATR) for longs or
            trough + (atr_mult * ATR) for shorts.
        direction: Position direction. 1 = long, -1 = short.

    Returns:
        Tuple of (exit_idx, exit_price):
        - exit_idx: Bar index where stop was triggered, or -1 if no exit.
        - exit_price: Close price at exit, or 0.0 if no exit.

    Note:
        - Uses high[i] for tracking long peaks, low[i] for short troughs
        - Exits are triggered on close[i], not intrabar
        - ATR must be pre-computed; this function does not calculate ATR

    Example:
        >>> import numpy as np
        >>> high = np.array([101.0, 102.0, 105.0, 104.0, 103.0, 98.0])
        >>> low = np.array([99.0, 100.0, 102.0, 101.0, 97.0, 95.0])
        >>> close = np.array([100.0, 101.0, 104.0, 102.0, 98.0, 96.0])
        >>> atr = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
        >>> # Long position with 2x ATR trailing stop
        >>> exit_idx, exit_price = atr_trailing_stop_nb(
        ...     high, low, close, atr,
        ...     entry_idx=0, atr_mult=2.0, direction=1
        ... )
    """
    n = len(close)

    # Edge case: no bars after entry
    if entry_idx >= n - 1:
        return (-1, 0.0)

    peak = close[entry_idx]

    for i in range(entry_idx + 1, n):
        if direction == 1:
            peak = max(peak, high[i])
            stop = peak - atr_mult * atr[i]
            if close[i] <= stop:
                return (i, close[i])
        else:
            peak = min(peak, low[i])
            stop = peak + atr_mult * atr[i]
            if close[i] >= stop:
                return (i, close[i])

    return (-1, 0.0)


def generate_trailing_exits(
    entries: NDArray[np.bool_],
    close: NDArray[np.float64],
    entry_prices: NDArray[np.float64],
    trail_percent: float,
    activation_percent: float = 0.0,
    direction: int = 1,
    high: NDArray[np.float64] | None = None,
    low: NDArray[np.float64] | None = None,
    atr: NDArray[np.float64] | None = None,
    atr_mult: float = 2.0,
    stop_type: Literal["delayed", "atr"] = "delayed",
) -> NDArray[np.bool_]:
    """Generate exit signal array from entry signals using trailing stops.

    High-level convenience wrapper that processes entry signals and generates
    corresponding exit signals compatible with VectorBT's from_signals().

    For each entry signal, this function:
    1. Finds the entry index and price
    2. Calls the appropriate JIT-compiled trailing stop function
    3. Marks the exit index in the output boolean array

    Args:
        entries: Boolean array of entry signals (True = entry occurred).
        close: Close price array (must match entries length).
        entry_prices: Price at each entry. Array same length as close with
            entry price filled at entry indices, 0.0 elsewhere.
        trail_percent: Trail distance as decimal (e.g., 0.02 = 2%).
        activation_percent: Required favorable move before trailing activates.
            Default 0.0 means trail immediately (standard trailing stop).
            Only used when stop_type="delayed".
        direction: 1 for long positions, -1 for short positions.
        high: High prices array. Required when stop_type="atr".
        low: Low prices array. Required when stop_type="atr".
        atr: Pre-computed ATR array. Required when stop_type="atr".
        atr_mult: ATR multiplier for stop distance. Only used when stop_type="atr".
        stop_type: "delayed" for delayed trailing stop, "atr" for ATR-based stop.

    Returns:
        Boolean array of exit signals aligned with close prices. True indicates
        an exit occurred at that bar.

    Raises:
        ValueError: If stop_type="atr" but high, low, or atr arrays are not provided.

    Example:
        >>> import numpy as np
        >>> # Setup data
        >>> close = np.array([100.0, 102.0, 105.0, 103.0, 100.0])
        >>> entries = np.array([True, False, False, False, False])
        >>> entry_prices = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        >>>
        >>> # Generate exits for long entries with 2% delayed trailing stop
        >>> exits = generate_trailing_exits(
        ...     entries, close, entry_prices,
        ...     trail_percent=0.02, activation_percent=0.01, direction=1
        ... )
        >>>
        >>> # Use with VectorBT
        >>> # pf = vbt.Portfolio.from_signals(
        >>> #     close, entries=entries, exits=exits, direction='long'
        >>> # )
    """
    n = len(close)
    exits = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return exits

    # Validate inputs for ATR stop
    if stop_type == "atr" and (high is None or low is None or atr is None):
        raise ValueError(
            "high, low, and atr arrays are required when stop_type='atr'"
        )

    # Find all entry indices
    entry_indices = np.where(entries)[0]

    for entry_idx in entry_indices:
        entry_price = entry_prices[entry_idx]

        if stop_type == "delayed":
            exit_idx, _exit_price, _peak = delayed_trailing_stop_nb(
                close,
                entry_price,
                entry_idx,
                trail_percent,
                activation_percent,
                direction,
            )
        else:  # stop_type == "atr"
            # Type narrowing - we validated these are not None above
            assert high is not None
            assert low is not None
            assert atr is not None
            exit_idx, _exit_price = atr_trailing_stop_nb(
                high,
                low,
                close,
                atr,
                entry_idx,
                atr_mult,
                direction,
            )

        if exit_idx >= 0:
            exits[exit_idx] = True

    return exits


__all__: list[str] = [
    "delayed_trailing_stop_nb",
    "atr_trailing_stop_nb",
    "generate_trailing_exits",
]
