"""Vectorized indicator calculations for fast parameter sweeps.

These functions accept arrays of periods and return multi-dimensional arrays
containing calculations for all period combinations simultaneously. This enables
calculating indicators once and testing many parameter combinations without
recalculation.

Example:
    >>> # Calculate Vortex for periods [14, 20] in one pass
    >>> vi_plus, vi_minus = vortex_indicator_vectorized(high, low, close, [14, 20])
    >>> vi_plus.shape  # (2, n_bars) - one row per period
    (2, 91144)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def weighted_moving_average_vectorized(
    data: NDArray[np.float64],
    periods: list[int] | NDArray[np.int32],
) -> NDArray[np.float64]:
    """Calculate WMA for multiple periods simultaneously.

    Args:
        data: Input data array (n_bars,)
        periods: List of WMA periods to calculate

    Returns:
        Array of shape (n_periods, n_bars) with WMA for each period
    """
    periods = np.asarray(periods, dtype=np.int32)
    n_periods = len(periods)
    n_bars = len(data)
    max_period = periods.max()

    result = np.full((n_periods, n_bars), np.nan, dtype=np.float64)

    for p_idx, period in enumerate(periods):
        weights = np.arange(1, period + 1, dtype=np.float64)
        weights = weights / weights.sum()

        for i in range(period - 1, n_bars):
            result[p_idx, i] = np.dot(data[i - period + 1 : i + 1], weights)

    return result


def hull_moving_average_vectorized(
    close: NDArray[np.float64],
    periods: list[int] | NDArray[np.int32],
) -> NDArray[np.float64]:
    """Calculate HMA for multiple periods simultaneously.

    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    Args:
        close: Close prices (n_bars,)
        periods: List of HMA periods to calculate

    Returns:
        Array of shape (n_periods, n_bars) with HMA for each period
    """
    periods = np.asarray(periods, dtype=np.int32)
    n_periods = len(periods)
    n_bars = len(close)

    result = np.full((n_periods, n_bars), np.nan, dtype=np.float64)

    for p_idx, period in enumerate(periods):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))

        # Calculate WMA(n/2) and WMA(n)
        wma_half = weighted_moving_average_vectorized(close, [half_period])[0]
        wma_full = weighted_moving_average_vectorized(close, [period])[0]

        # Raw HMA = 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # Final HMA = WMA(raw_hma, sqrt(n))
        hma = weighted_moving_average_vectorized(raw_hma, [sqrt_period])[0]
        result[p_idx] = hma

    return result


def vortex_indicator_vectorized(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    periods: list[int] | NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate Vortex Indicator for multiple periods simultaneously.

    Args:
        high: High prices (n_bars,)
        low: Low prices (n_bars,)
        close: Close prices (n_bars,)
        periods: List of VI periods to calculate

    Returns:
        Tuple of (VI+, VI-) arrays, each of shape (n_periods, n_bars)
    """
    periods = np.asarray(periods, dtype=np.int32)
    n_periods = len(periods)
    n_bars = len(close)
    max_period = periods.max()

    # Calculate True Range (same for all periods)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = np.nan

    # Calculate Vortex Movement (same for all periods)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan

    # Calculate rolling sums for each period
    vi_plus = np.full((n_periods, n_bars), np.nan, dtype=np.float64)
    vi_minus = np.full((n_periods, n_bars), np.nan, dtype=np.float64)

    for p_idx, period in enumerate(periods):
        for i in range(period, n_bars):
            sum_vm_plus = np.nansum(vm_plus[i - period + 1 : i + 1])
            sum_vm_minus = np.nansum(vm_minus[i - period + 1 : i + 1])
            sum_tr = np.nansum(tr[i - period + 1 : i + 1])

            if sum_tr > 0:
                vi_plus[p_idx, i] = sum_vm_plus / sum_tr
                vi_minus[p_idx, i] = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


def calculate_atr_vectorized(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    periods: list[int] | NDArray[np.int32],
) -> NDArray[np.float64]:
    """Calculate ATR for multiple periods simultaneously.

    Args:
        high: High prices (n_bars,)
        low: Low prices (n_bars,)
        close: Close prices (n_bars,)
        periods: List of ATR periods to calculate

    Returns:
        Array of shape (n_periods, n_bars) with ATR for each period
    """
    periods = np.asarray(periods, dtype=np.int32)
    n_periods = len(periods)
    n_bars = len(close)

    # Calculate True Range (same for all periods)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = np.nan

    # Calculate rolling average for each period
    atr = np.full((n_periods, n_bars), np.nan, dtype=np.float64)

    for p_idx, period in enumerate(periods):
        for i in range(period, n_bars):
            atr[p_idx, i] = np.nanmean(tr[i - period + 1 : i + 1])

    return atr


__all__ = [
    "weighted_moving_average_vectorized",
    "hull_moving_average_vectorized",
    "vortex_indicator_vectorized",
    "calculate_atr_vectorized",
]
