#!/usr/bin/env python3
"""Verification script for I2.T1 implementation.

Tests all components of the JIT utilities and bar factory infrastructure.
"""

import sys
import numpy as np
from typing import Any

# Import the modules we're testing
from simple_futures_backtester.utils import (
    get_njit_decorator,
    ensure_float64,
    ensure_int64,
    preallocate_array,
    preallocate_2d_array,
    is_contiguous,
    validate_ohlcv_arrays,
)
from simple_futures_backtester.bars import (
    BarSeries,
    register_bar_type,
    get_bar_generator,
    list_bar_types,
    unregister_bar_type,
    clear_bar_registry,
)


def test_njit_decorator():
    """Test get_njit_decorator functionality."""
    print("\n=== Testing JIT Decorator ===")

    # Test basic decorator creation
    jit = get_njit_decorator()
    print(f"✓ Created default decorator: {jit}")

    # Test with custom options
    jit_parallel = get_njit_decorator(parallel=True)
    print(f"✓ Created parallel decorator: {jit_parallel}")

    # Test decoration of a function
    @get_njit_decorator(cache=False)
    def add_arrays(a, b):
        result = np.empty_like(a)
        for i in range(len(a)):
            result[i] = a[i] + b[i]
        return result

    # Test execution
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([4.0, 5.0, 6.0])
    result = add_arrays(arr1, arr2)
    expected = np.array([5.0, 7.0, 9.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ JIT-compiled function executed correctly: {result}")


def test_ensure_float64():
    """Test ensure_float64 dtype conversion."""
    print("\n=== Testing ensure_float64 ===")

    # Test int32 to float64
    arr_int32 = np.array([100, 101, 102], dtype=np.int32)
    result = ensure_float64(arr_int32)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
    assert result.flags['C_CONTIGUOUS'], "Expected C-contiguous array"
    print(f"✓ Converted int32 to float64: {result.dtype}")

    # Test float64 identity (no copy)
    arr_float64 = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    result = ensure_float64(arr_float64)
    assert result.dtype == np.float64
    assert result.flags['C_CONTIGUOUS']
    print(f"✓ Float64 identity preserved: {result.dtype}")

    # Test error on invalid input
    try:
        ensure_float64("not an array")  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Correctly raised TypeError: {e}")


def test_ensure_int64():
    """Test ensure_int64 dtype conversion."""
    print("\n=== Testing ensure_int64 ===")

    # Test float64 to int64
    arr_float = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    result = ensure_int64(arr_float)
    assert result.dtype == np.int64, f"Expected int64, got {result.dtype}"
    assert result.flags['C_CONTIGUOUS'], "Expected C-contiguous array"
    print(f"✓ Converted float64 to int64: {result.dtype}")

    # Test int64 identity
    arr_int64 = np.array([1000, 2000, 3000], dtype=np.int64)
    result = ensure_int64(arr_int64)
    assert result.dtype == np.int64
    print(f"✓ Int64 identity preserved: {result.dtype}")

    # Test NaN handling
    arr_nan = np.array([1.0, np.nan, 3.0])
    try:
        ensure_int64(arr_nan)
        assert False, "Should have raised ValueError for NaN"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for NaN: {e}")


def test_preallocate_array():
    """Test preallocate_array functionality."""
    print("\n=== Testing preallocate_array ===")

    # Test basic allocation
    arr = preallocate_array(100)
    assert arr.shape == (100,), f"Expected shape (100,), got {arr.shape}"
    assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"
    assert arr.flags['C_CONTIGUOUS'], "Expected C-contiguous array"
    assert np.all(np.isnan(arr)), "Expected all NaN values for float64"
    print(f"✓ Preallocated float64 array: shape={arr.shape}, dtype={arr.dtype}")

    # Test int64 allocation
    arr_int = preallocate_array(100, dtype=np.int64)
    assert arr_int.dtype == np.int64
    assert np.all(arr_int == 0), "Expected all zero values for int64"
    print(f"✓ Preallocated int64 array: shape={arr_int.shape}, dtype={arr_int.dtype}")

    # Test custom fill value
    arr_fill = preallocate_array(100, fill_value=42.0)
    assert np.all(arr_fill == 42.0), "Expected all 42.0 values"
    print(f"✓ Preallocated with custom fill value: {arr_fill[0]}")

    # Test negative size error
    try:
        preallocate_array(-10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for negative size: {e}")


def test_preallocate_2d_array():
    """Test preallocate_2d_array functionality."""
    print("\n=== Testing preallocate_2d_array ===")

    arr = preallocate_2d_array(100, 4)
    assert arr.shape == (100, 4), f"Expected shape (100, 4), got {arr.shape}"
    assert arr.dtype == np.float64
    assert arr.flags['C_CONTIGUOUS']
    print(f"✓ Preallocated 2D array: shape={arr.shape}, dtype={arr.dtype}")


def test_is_contiguous():
    """Test is_contiguous check."""
    print("\n=== Testing is_contiguous ===")

    arr = np.array([1, 2, 3, 4, 5])
    assert is_contiguous(arr), "Expected contiguous array"
    print("✓ Contiguous array detected correctly")

    # Strided view is not contiguous
    strided = arr[::2]
    assert not is_contiguous(strided), "Expected non-contiguous strided view"
    print("✓ Non-contiguous strided view detected correctly")


def test_validate_ohlcv_arrays():
    """Test validate_ohlcv_arrays."""
    print("\n=== Testing validate_ohlcv_arrays ===")

    # Create test arrays
    open_arr = np.array([100.0, 101.0, 102.0])
    high_arr = np.array([102.0, 103.0, 104.0])
    low_arr = np.array([99.0, 100.0, 101.0])
    close_arr = np.array([101.0, 102.0, 103.0])
    volume_arr = np.array([1000, 2000, 3000])

    o, h, l, c, v = validate_ohlcv_arrays(open_arr, high_arr, low_arr, close_arr, volume_arr)

    assert o.dtype == np.float64
    assert h.dtype == np.float64
    assert l.dtype == np.float64
    assert c.dtype == np.float64
    assert v.dtype == np.int64
    print("✓ OHLCV arrays validated and normalized")

    # Test inconsistent lengths
    bad_volume = np.array([1000, 2000])
    try:
        validate_ohlcv_arrays(open_arr, high_arr, low_arr, close_arr, bad_volume)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for inconsistent lengths: {e}")


def test_bar_series():
    """Test BarSeries dataclass."""
    print("\n=== Testing BarSeries ===")

    # Create a BarSeries
    bars = BarSeries(
        type="test",
        parameters={"param1": 10},
        open=np.array([100.0, 110.0], dtype=np.float32),  # Test dtype conversion
        high=np.array([110.0, 120.0]),
        low=np.array([100.0, 110.0]),
        close=np.array([110.0, 120.0]),
        volume=np.array([1000.0, 2000.0]),  # Test dtype conversion
        index_map=np.array([50, 120]),
    )

    assert bars.type == "test"
    assert bars.parameters == {"param1": 10}
    assert len(bars) == 2
    assert not bars.is_empty
    assert bars.open.dtype == np.float64
    assert bars.volume.dtype == np.int64
    assert bars.index_map.dtype == np.int64
    print(f"✓ BarSeries created successfully: type={bars.type}, len={len(bars)}")

    # Test empty BarSeries
    empty_bars = BarSeries(type="empty")
    assert len(empty_bars) == 0
    assert empty_bars.is_empty
    print("✓ Empty BarSeries works correctly")

    # Test inconsistent lengths
    try:
        bad_bars = BarSeries(
            type="bad",
            open=np.array([100.0, 110.0]),
            high=np.array([110.0]),  # Wrong length
            low=np.array([100.0, 110.0]),
            close=np.array([110.0, 120.0]),
            volume=np.array([1000, 2000]),
            index_map=np.array([50, 120]),
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for inconsistent lengths: {e}")


def test_bar_factory_registry():
    """Test bar factory registry."""
    print("\n=== Testing Bar Factory Registry ===")

    # Clear registry first
    clear_bar_registry()
    assert list_bar_types() == []
    print("✓ Registry cleared")

    # Register a test generator
    def test_generator(open, high, low, close, volume, **kwargs):
        return BarSeries(
            type="test_bars",
            parameters=kwargs,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            index_map=np.arange(len(open), dtype=np.int64),
        )

    register_bar_type("test_bars", test_generator)
    assert "test_bars" in list_bar_types()
    print("✓ Registered test_bars generator")

    # Retrieve generator
    generator = get_bar_generator("test_bars")
    assert generator == test_generator
    print("✓ Retrieved test_bars generator")

    # Test generator execution
    o = np.array([100.0, 101.0])
    h = np.array([102.0, 103.0])
    l = np.array([99.0, 100.0])
    c = np.array([101.0, 102.0])
    v = np.array([1000, 2000])

    bars = generator(o, h, l, c, v, param1=10)
    assert bars.type == "test_bars"
    assert bars.parameters == {"param1": 10}
    assert len(bars) == 2
    print("✓ Generator executed successfully")

    # Test unknown bar type
    try:
        get_bar_generator("unknown")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✓ Correctly raised KeyError for unknown type: {e}")

    # Test empty name registration
    try:
        register_bar_type("", test_generator)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for empty name: {e}")

    # Test non-callable registration
    try:
        register_bar_type("bad", "not_callable")  # type: ignore
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for non-callable: {e}")

    # Test unregister
    assert unregister_bar_type("test_bars")
    assert not unregister_bar_type("test_bars")  # Already unregistered
    print("✓ Unregister works correctly")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("VERIFYING I2.T1 IMPLEMENTATION")
    print("=" * 60)

    try:
        test_njit_decorator()
        test_ensure_float64()
        test_ensure_int64()
        test_preallocate_array()
        test_preallocate_2d_array()
        test_is_contiguous()
        test_validate_ohlcv_arrays()
        test_bar_series()
        test_bar_factory_registry()

        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ VERIFICATION FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
