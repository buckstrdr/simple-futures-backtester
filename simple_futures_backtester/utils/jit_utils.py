"""JIT compilation utilities for high-performance Numba kernels.

Provides shared decorators, dtype conversion helpers, and array pre-allocation
utilities to ensure consistent JIT-safety across all bar generators and indicators.

All utilities enforce:
- C-contiguous array layouts for optimal Numba performance
- float64 dtype for prices (open, high, low, close)
- int64 dtype for indices and volume

Performance Note:
    cache=True is enabled by default for production use. During development,
    set cache=False to avoid stale cached versions when modifying JIT functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import DTypeLike, NDArray

# Type variable for generic JIT-decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def get_njit_decorator(
    cache: bool = True,
    parallel: bool = False,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Get configured Numba @njit decorator with project defaults.

    Creates a Numba JIT decorator with sensible defaults for high-performance
    bar generation and indicator computation. The returned decorator compiles
    Python functions to optimized machine code.

    Args:
        cache: Enable disk caching of compiled functions. Default True for
            production use to avoid recompilation overhead. Set False during
            development when iterating on JIT function implementations.
        parallel: Enable automatic parallelization with Numba's prange.
            Default False; must be explicitly opted-in since not all algorithms
            are parallelizable and incorrect use can cause race conditions.
        **kwargs: Additional arguments passed directly to numba.njit().
            Common options include:
            - nogil: bool - Release GIL (default False)
            - fastmath: bool - Enable unsafe math optimizations (default False)
            - boundscheck: bool - Enable bounds checking (default varies)

    Returns:
        Configured @njit decorator ready to apply to functions.

    Example:
        >>> jit = get_njit_decorator(cache=True, fastmath=True)
        >>> @jit
        ... def compute_renko(prices, brick_size):
        ...     # JIT-compiled implementation
        ...     pass
    """
    return njit(cache=cache, parallel=parallel, **kwargs)  # type: ignore[no-any-return]


def ensure_float64(arr: NDArray[Any]) -> NDArray[np.float64]:
    """Convert array to float64 dtype, ensuring JIT safety and C-contiguity.

    Ensures the input array is suitable for Numba JIT-compiled functions by
    converting to float64 and guaranteeing C-contiguous memory layout. This is
    essential for price data (open, high, low, close) in bar generators.

    Args:
        arr: Input numpy array of any numeric dtype.

    Returns:
        Array converted to float64 with C-contiguous memory layout.
        If the input is already float64 and C-contiguous, returns a view
        (no copy) for performance.

    Raises:
        ValueError: If array cannot be converted to float64 (e.g., contains
            non-numeric data that cannot be coerced).
        TypeError: If input is not a numpy array.

    Example:
        >>> prices = np.array([100, 101, 102], dtype=np.int32)
        >>> float_prices = ensure_float64(prices)
        >>> float_prices.dtype
        dtype('float64')
        >>> float_prices.flags['C_CONTIGUOUS']
        True
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr).__name__}")

    if arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]:
        return arr

    try:
        return np.ascontiguousarray(arr, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert array to float64: {e}") from e


def ensure_int64(arr: NDArray[Any]) -> NDArray[np.int64]:
    """Convert array to int64 dtype, ensuring JIT safety and C-contiguity.

    Ensures the input array is suitable for Numba JIT-compiled functions by
    converting to int64 and guaranteeing C-contiguous memory layout. This is
    essential for volume data and index mappings in bar generators.

    Args:
        arr: Input numpy array of any numeric dtype.

    Returns:
        Array converted to int64 with C-contiguous memory layout.
        If the input is already int64 and C-contiguous, returns a view
        (no copy) for performance.

    Raises:
        ValueError: If array cannot be converted to int64 (e.g., contains
            non-numeric data, NaN values, or values outside int64 range).
        TypeError: If input is not a numpy array.

    Note:
        Float arrays with NaN or Inf values will raise an error during
        conversion since int64 cannot represent these values.

    Example:
        >>> volume = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
        >>> int_volume = ensure_int64(volume)
        >>> int_volume.dtype
        dtype('int64')
        >>> int_volume.flags['C_CONTIGUOUS']
        True
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr).__name__}")

    if arr.dtype == np.int64 and arr.flags["C_CONTIGUOUS"]:
        return arr

    if np.issubdtype(arr.dtype, np.floating) and np.any(~np.isfinite(arr)):
        raise ValueError("Cannot convert array with NaN/Inf values to int64")

    try:
        return np.ascontiguousarray(arr, dtype=np.int64)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(f"Cannot convert array to int64: {e}") from e


def preallocate_array(
    size: int,
    dtype: DTypeLike = np.float64,
    fill_value: float | int | None = None,
) -> NDArray[Any]:
    """Preallocate C-contiguous numpy array with specified size and dtype.

    Creates a pre-allocated array for use in JIT-compiled functions. Using
    pre-allocated arrays avoids repeated memory allocation inside hot loops,
    which is critical for achieving 1M+ rows/sec throughput.

    Args:
        size: Number of elements to allocate. Must be non-negative.
        dtype: NumPy dtype for the array. Default float64 for price data.
            Use np.int64 for volume or index data.
        fill_value: Initial value for all elements. If None, uses NaN for
            floating-point dtypes and 0 for integer dtypes.

    Returns:
        C-contiguous numpy array of specified size and dtype, initialized
        with the fill value.

    Raises:
        ValueError: If size is negative.

    Example:
        >>> prices = preallocate_array(1000, np.float64)
        >>> prices.shape
        (1000,)
        >>> prices.flags['C_CONTIGUOUS']
        True
        >>> np.all(np.isnan(prices))
        True

        >>> indices = preallocate_array(1000, np.int64)
        >>> indices[0]
        0
    """
    if size < 0:
        raise ValueError(f"Size must be non-negative, got {size}")

    arr: NDArray[Any] = np.empty(size, dtype=dtype, order="C")

    if fill_value is not None:
        arr.fill(fill_value)
    elif np.issubdtype(arr.dtype, np.floating):
        arr.fill(np.nan)
    else:
        arr.fill(0)

    return arr


def preallocate_2d_array(
    rows: int,
    cols: int,
    dtype: DTypeLike = np.float64,
    fill_value: float | int | None = None,
) -> NDArray[Any]:
    """Preallocate C-contiguous 2D numpy array with specified shape and dtype.

    Creates a pre-allocated 2D array for use in JIT-compiled functions that
    need to output multiple time series (e.g., OHLCV bar arrays).

    Args:
        rows: Number of rows to allocate. Must be non-negative.
        cols: Number of columns to allocate. Must be non-negative.
        dtype: NumPy dtype for the array. Default float64 for price data.
        fill_value: Initial value for all elements. If None, uses NaN for
            floating-point dtypes and 0 for integer dtypes.

    Returns:
        C-contiguous 2D numpy array of specified shape and dtype.

    Raises:
        ValueError: If rows or cols is negative.

    Example:
        >>> ohlc = preallocate_2d_array(1000, 4, np.float64)
        >>> ohlc.shape
        (1000, 4)
        >>> ohlc.flags['C_CONTIGUOUS']
        True
    """
    if rows < 0:
        raise ValueError(f"Rows must be non-negative, got {rows}")
    if cols < 0:
        raise ValueError(f"Cols must be non-negative, got {cols}")

    arr: NDArray[Any] = np.empty((rows, cols), dtype=dtype, order="C")

    if fill_value is not None:
        arr.fill(fill_value)
    elif np.issubdtype(arr.dtype, np.floating):
        arr.fill(np.nan)
    else:
        arr.fill(0)

    return arr


def is_contiguous(arr: NDArray[Any]) -> bool:
    """Check if array is C-contiguous for JIT safety.

    C-contiguous arrays have optimal memory layout for Numba JIT functions.
    Use this to validate arrays before passing to performance-critical code.

    Args:
        arr: NumPy array to check.

    Returns:
        True if array is C-contiguous, False otherwise.

    Example:
        >>> arr = np.array([1, 2, 3])
        >>> is_contiguous(arr)
        True
        >>> is_contiguous(arr[::2])  # Strided view is not contiguous
        False
    """
    return bool(arr.flags["C_CONTIGUOUS"])


def validate_ohlcv_arrays(
    open_arr: NDArray[Any],
    high_arr: NDArray[Any],
    low_arr: NDArray[Any],
    close_arr: NDArray[Any],
    volume_arr: NDArray[Any],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Validate and normalize OHLCV arrays for JIT-compiled bar generators.

    Ensures all input arrays have consistent lengths and proper dtypes for
    use in Numba JIT-compiled functions. Converts price arrays to float64
    and volume to int64.

    Args:
        open_arr: Opening prices array.
        high_arr: High prices array.
        low_arr: Low prices array.
        close_arr: Closing prices array.
        volume_arr: Volume array.

    Returns:
        Tuple of (open, high, low, close, volume) arrays with normalized
        dtypes and C-contiguous memory layout.

    Raises:
        ValueError: If arrays have inconsistent lengths.
        TypeError: If any input is not a numpy array.

    Example:
        >>> o = np.array([100.0, 101.0])
        >>> h = np.array([102.0, 103.0])
        >>> l = np.array([99.0, 100.0])
        >>> c = np.array([101.0, 102.0])
        >>> v = np.array([1000, 2000])
        >>> o, h, l, c, v = validate_ohlcv_arrays(o, h, l, c, v)
    """
    lengths = {
        "open": len(open_arr),
        "high": len(high_arr),
        "low": len(low_arr),
        "close": len(close_arr),
        "volume": len(volume_arr),
    }

    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(f"OHLCV arrays have inconsistent lengths: {lengths}")

    return (
        ensure_float64(open_arr),
        ensure_float64(high_arr),
        ensure_float64(low_arr),
        ensure_float64(close_arr),
        ensure_int64(volume_arr),
    )
