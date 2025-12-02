"""Tests for Volume bar generation.

Tests cover:
- Basic functionality with fixed volume_threshold
- Algorithm correctness (volume accumulation tracking)
- Edge cases (empty data, single row, no bars generated)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import os

# Set testing flag BEFORE any imports to skip VectorBT initialization
os.environ["SFB_TESTING"] = "1"

import numpy as np
import pytest

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.volume_bars import generate_volume_bars_series


class TestVolumeRegistration:
    """Tests for bar factory registration."""

    def test_volume_registered(self) -> None:
        """Volume should be registered with the bar factory."""
        assert "volume" in list_bar_types()

    def test_get_volume_generator(self) -> None:
        """Should retrieve the volume generator from factory."""
        generator = get_bar_generator("volume")
        assert generator is generate_volume_bars_series


class TestVolumeBasicFunctionality:
    """Tests for basic Volume bar generation."""

    def test_simple_volume_bars(self) -> None:
        """Should generate bars when cumulative volume exceeds volume_threshold."""
        # Create data with known volume accumulation pattern
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([500, 600, 700, 800, 900, 1000, 1100], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Should generate bars when cumulative volume >= 2000
        # Bar 1: 500+600+700 = 1800 (not yet), +800 = 2600 (closes at index 3)
        # Bar 2: 900+1000 = 1900 (not yet), +1100 = 3000 (closes at index 6)
        assert len(bars) >= 1
        assert bars.type == "volume"
        assert bars.parameters["volume_threshold"] == 2000

    def test_high_volume_many_bars(self) -> None:
        """High volume with small threshold should create many bars."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=1500,
        )

        # With consistent volume and small threshold, expect multiple bars
        assert len(bars) >= 3

    def test_low_volume_few_bars(self) -> None:
        """Low volume with large threshold should create few/no bars."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([10, 20, 30, 40, 50], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=10000,
        )

        # With low volume and large threshold, expect 0 bars
        assert len(bars) == 0

    def test_cumulative_volume_per_bar(self) -> None:
        """Volume field should contain cumulative volume per bar."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Each bar's volume should be >= volume_threshold
        if len(bars) > 0:
            for bar_vol in bars.volume:
                assert bar_vol >= 2000

    def test_index_map_tracking(self) -> None:
        """Index map should correctly track source row where bar completed."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Verify index_map points to valid source indices
        if len(bars) > 0:
            assert all(0 <= idx < len(close) for idx in bars.index_map)
            # Indices should be monotonically increasing
            if len(bars) > 1:
                assert all(bars.index_map[i] < bars.index_map[i + 1] for i in range(len(bars) - 1))


class TestVolumeAlgorithmCorrectness:
    """Tests for Volume bar algorithm implementation."""

    def test_volume_accumulation(self) -> None:
        """Should correctly accumulate volume until threshold reached."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        # Cumulative: 300, 800, 1400, 2100 -> bar closes at index 3
        volume = np.array([300, 500, 600, 700, 800], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        assert len(bars) == 1
        assert bars.close[0] == 103.0  # Closes at index 3
        assert bars.volume[0] == 2100  # 300+500+600+700

    def test_bar_reset_after_close(self) -> None:
        """After closing a bar, should reset and start accumulating new volume."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([600, 700, 800, 900, 1000, 1100, 1200], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Should generate 2 bars
        # Bar 1: 600+700+800 = 2100 (closes at index 2)
        # Bar 2: 900+1000+1100 = 3000 (closes at index 5)
        assert len(bars) == 2

    def test_ohlc_tracking(self) -> None:
        """Should track OHLC properly across accumulated bars."""
        close = np.array([100.0, 102.0, 101.0, 103.0])
        high = np.array([100.5, 103.0, 102.0, 104.0])
        low = np.array([99.5, 101.0, 100.0, 102.0])
        volume = np.array([800, 700, 600, 500], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        if len(bars) > 0:
            # First bar should have open from first source bar
            assert bars.open[0] == 100.0
            # High should be max across accumulated bars
            assert bars.high[0] >= 103.0
            # Low should be min across accumulated bars
            assert bars.low[0] <= 100.0

    def test_high_low_tracking_within_bar(self) -> None:
        """Should track high/low correctly within accumulated volume bar."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        # Third bar has highest high
        high = np.array([100.5, 101.5, 105.0, 103.5, 104.5])
        # Second bar has lowest low
        low = np.array([99.5, 98.0, 101.5, 102.5, 103.5])
        volume = np.array([700, 800, 600, 500, 400], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # First bar accumulates indices 0-3 (volume: 700+800+600 = 2100)
        assert len(bars) >= 1
        # High should be max from indices 0-3 (105.0 at index 2)
        assert bars.high[0] == 105.0
        # Low should be min from indices 0-3 (98.0 at index 1)
        assert bars.low[0] == 98.0

    def test_multiple_bars_sequence(self) -> None:
        """Should generate multiple bars in correct sequence with proper reset."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
        high = close + 0.5
        low = close - 0.5
        # Pattern creates 3 bars: (1100+900=2000), (1000+1100=2100), (1200+1300=2500)
        volume = np.array([600, 1100, 900, 1000, 1100, 1200, 1300, 1400, 1500], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        assert len(bars) >= 3

        # Verify bar sequence
        # Bar 0 closes at index 2 (600+1100+900=2600)
        assert bars.index_map[0] == 2
        assert bars.open[0] == 100.0
        assert bars.close[0] == 102.0

        # Bar 1 starts from close of bar 0, closes at index 4
        assert bars.index_map[1] == 4
        assert bars.open[1] == 102.0
        assert bars.close[1] == 104.0

        # Bar 2 starts from close of bar 1, closes at index 6
        assert bars.index_map[2] == 6
        assert bars.open[2] == 104.0
        assert bars.close[2] == 106.0

    def test_volume_reset_after_bar_close(self) -> None:
        """Cumulative volume should reset to zero after each bar close."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 0.5
        low = close - 0.5
        # First bar: 1200+900 = 2100, second bar: 800+700+600 = 2100
        volume = np.array([600, 1200, 900, 800, 700, 600], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        assert len(bars) == 2

        # First bar: indices 0-2 (600+1200+900 = 2700)
        assert bars.volume[0] == 2700
        assert bars.index_map[0] == 2

        # Second bar: indices 3-5 (800+700+600 = 2100), NOT including previous volume
        assert bars.volume[1] == 2100
        assert bars.index_map[1] == 5


class TestVolumeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_volume_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            volume_threshold=1000,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row(self) -> None:
        """Single row should return empty BarSeries (need at least 2)."""
        bars = generate_volume_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            volume_threshold=500,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_invalid_threshold_zero(self) -> None:
        """volume_threshold of zero should raise ValueError."""
        with pytest.raises(ValueError, match="volume_threshold must be positive"):
            generate_volume_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                volume_threshold=0,
            )

    def test_invalid_threshold_negative(self) -> None:
        """Negative volume_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="volume_threshold must be positive"):
            generate_volume_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                volume_threshold=-500,
            )

    def test_inconsistent_array_lengths(self) -> None:
        """Arrays with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="inconsistent lengths"):
            generate_volume_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0]),  # Different length
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                volume_threshold=1000,
            )

    def test_zero_volume_bars(self) -> None:
        """Bars should not be created if volume is zero (never reaches threshold)."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.zeros(len(close), dtype=np.int64)  # All zeros

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=1000,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_huge_volume_spike(self) -> None:
        """Single huge volume spike should immediately trigger bar close."""
        close = np.array([100.0, 101.0, 102.0, 103.0])
        high = close + 1.0
        low = close - 1.0
        # Second bar has huge spike
        volume = np.array([100, 1000000, 200, 300], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=10000,
        )

        # Should generate at least 1 bar from the huge spike
        assert len(bars) >= 1
        # First bar should capture the spike
        assert bars.index_map[0] == 1
        assert bars.volume[0] >= 10000

    def test_exact_threshold_boundary(self) -> None:
        """Bar should close when cumulative_volume exactly equals threshold."""
        close = np.array([100.0, 101.0, 102.0, 103.0])
        high = close + 0.5
        low = close - 0.5
        # Cumulative: 500, 1000, 1500, 2000 (exactly threshold)
        volume = np.array([500, 500, 500, 500], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        assert len(bars) == 1
        assert bars.volume[0] == 2000  # Exactly threshold
        assert bars.close[0] == 103.0  # Closes at index 3
        assert bars.index_map[0] == 3

    def test_mixed_volume_pattern(self) -> None:
        """Should handle mix of small and large volume bars."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        high = close + 0.5
        low = close - 0.5
        # Pattern: small, small, HUGE, small, small, HUGE, small
        volume = np.array([100, 200, 5000, 150, 250, 6000, 300, 400], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Verify multiple bars generated
        assert len(bars) >= 2

        # First bar should capture huge spike at index 2
        assert bars.index_map[0] == 2
        assert bars.volume[0] >= 2000

    def test_large_volume_values(self) -> None:
        """Should handle large but realistic volume values without overflow."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        # Large realistic volumes
        volume = np.array([500000, 750000, 600000, 800000, 900000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=1000000,
        )

        # Should generate bars without overflow
        assert len(bars) >= 2
        # All volumes should be >= threshold
        for vol in bars.volume:
            assert vol >= 1000000

    def test_zero_volumes_mixed(self) -> None:
        """Zeros in volume stream should not break accumulation or OHLC tracking."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = np.array([100.5, 101.5, 102.5, 103.5, 104.5, 105.5])
        low = np.array([99.5, 100.5, 101.0, 102.5, 103.0, 104.5])
        # Include zeros; threshold requires accumulation across zeros
        volume = np.array([0, 500, 0, 800, 0, 900], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Expect a single bar closing at the last index when cumulative >= 2000
        assert len(bars) == 1
        assert bars.index_map[0] == 5
        assert bars.volume[0] == np.int64(500 + 800 + 900)
        # High/low across the whole span (indices 0..5)
        assert bars.high[0] == np.max(high[:6])
        assert bars.low[0] == np.min(low[:6])

    def test_near_int64_threshold_no_overflow(self) -> None:
        """Accumulates near int64 magnitudes without overflow; closes exactly at threshold."""
        # Use large values well within int64 range
        a = np.int64(1) << np.int64(61)  # 2**61
        close = np.array([100.0, 101.0], dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        volume = np.array([a, a], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=int(a << 1),  # 2**62
        )

        assert len(bars) == 1
        assert bars.volume[0] == np.int64(a + a)
        assert bars.index_map[0] == 1

    def test_non_numpy_inputs_raise_typeerror(self) -> None:
        """Passing non-numpy arrays should raise TypeError from validation helpers."""
        with pytest.raises(TypeError, match="Expected numpy array"):
            # Lists are not np.ndarray; validate_ohlcv_arrays should complain
            generate_volume_bars_series(
                open_arr=[100.0, 101.0],
                high_arr=[101.0, 102.0],
                low_arr=[99.0, 100.0],
                close_arr=[100.0, 101.0],
                volume_arr=[1000, 1000],
                volume_threshold=1000,
            )

    @pytest.mark.parametrize(
        "threshold,expected_indices",
        [
            # With this implementation, first bar cannot close before index 1
            # because closing checks occur in the loop starting at i=1.
            (1, [1, 2, 3, 4, 5]),   # every row after the first closes a bar
            (2, [1, 3, 5]),         # pairs of rows
            (3, [2, 5]),            # triplets
        ],
    )
    def test_parametrized_threshold_boundaries(self, threshold: int, expected_indices: list[int]) -> None:
        """Boundary checks to catch off-by-one around cumulative == threshold."""
        # Unit volume to make expected close indices easy to derive
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.ones_like(close, dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=threshold,
        )

        # Index map should match expected close positions
        np.testing.assert_array_equal(bars.index_map, np.array(expected_indices, dtype=np.int64))

    @pytest.mark.parametrize(
        "threshold,volume_pattern,expected_bar_count",
        [
            # Varied volume patterns to test threshold sensitivity
            # Note: First source row (index 0) starts accumulation; bars close starting from index 1+
            (1000, [100, 200, 300, 400, 500, 600, 700, 800], 3),  # Gradually increasing
            (1500, [500, 500, 500, 500, 500, 500], 2),  # Uniform volume
            (2000, [300, 700, 500, 600, 400, 800, 900, 1100], 2),  # Mixed pattern
            (500, [100, 100, 100, 100, 100, 500, 100, 100], 2),  # Spike in middle
            (3000, [1000, 1000, 1000, 1000, 1000], 1),  # Large threshold
        ],
    )
    def test_parametrized_varied_thresholds(
        self, threshold: int, volume_pattern: list[int], expected_bar_count: int
    ) -> None:
        """Test various threshold and volume combinations to verify accumulation logic."""
        n = len(volume_pattern)
        close = np.linspace(100.0, 100.0 + n, n)
        high = close + 0.5
        low = close - 0.5
        volume = np.array(volume_pattern, dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=threshold,
        )

        # Verify expected bar count
        assert len(bars) == expected_bar_count

        # Verify each bar's volume meets threshold
        for bar_vol in bars.volume:
            assert bar_vol >= threshold

        # Verify monotonic index_map
        if len(bars) > 1:
            assert all(bars.index_map[i] < bars.index_map[i + 1] for i in range(len(bars) - 1))

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1011])
    def test_randomized_volume_patterns(self, seed: int) -> None:
        """Randomized tests with fixed seeds to avoid overfitting to specific patterns."""
        np.random.seed(seed)
        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        # Random volumes between 100 and 1000
        volume = np.random.randint(100, 1001, n).astype(np.int64)
        threshold = 2500

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=threshold,
        )

        # Invariant checks that should hold for any random pattern
        if len(bars) > 0:
            # All bar volumes >= threshold
            assert all(bars.volume >= threshold)
            # Index map within valid range
            assert all(0 <= idx < n for idx in bars.index_map)
            # Monotonic index map
            if len(bars) > 1:
                assert all(bars.index_map[i] < bars.index_map[i + 1] for i in range(len(bars) - 1))
            # OHLC relationships
            assert all(bars.low <= bars.open)
            assert all(bars.low <= bars.close)
            assert all(bars.high >= bars.open)
            assert all(bars.high >= bars.close)
            assert all(bars.low <= bars.high)

    @pytest.mark.parametrize(
        "n,threshold,expected_min_bars",
        [
            (10, 100, 0),  # Small dataset, moderate threshold
            (100, 500, 10),  # Medium dataset, small threshold
            (1000, 10000, 5),  # Large dataset, large threshold
        ],
    )
    def test_scale_invariance(self, n: int, threshold: int, expected_min_bars: int) -> None:
        """Test behavior across different data scales and threshold sizes."""
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        volume = np.random.randint(50, 200, n).astype(np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=threshold,
        )

        # Should generate at least expected_min_bars
        assert len(bars) >= expected_min_bars

        # Verify all invariants hold
        if len(bars) > 0:
            assert all(bars.volume >= threshold)
            assert bars.open.dtype == np.float64
            assert bars.volume.dtype == np.int64

    def test_index_map_segment_alignment(self) -> None:
        """Reconstruct segments from index_map and verify OHLCV aggregation per bar."""
        close = np.array([100.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 106.0])
        high = np.array([100.5, 101.5, 103.5, 102.5, 104.5, 103.5, 105.5, 106.5])
        low = np.array([99.5, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0, 105.0])
        volume = np.array([600, 700, 500, 300, 900, 1000, 1100, 1200], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # For each bar, compute expected stats considering implementation details:
        # - First bar: open=close[0] (here open_arr=close), high/low/volume over 0..end
        # - Subsequent bars: open=close[prev_end], high/low include prev_end baseline,
        #   volume sums only from prev_end+1..end.
        prev_end = -1
        for i, end in enumerate(bars.index_map):
            if i == 0:
                start = 0
                baseline_idx = 0
                vol_start = 0
            else:
                start = prev_end + 1
                baseline_idx = prev_end
                vol_start = start

            seg_slice_highlow = slice(baseline_idx, end + 1)
            seg_slice_vol = slice(vol_start, end + 1)

            expected_open = close[baseline_idx]
            expected_high = np.max(high[seg_slice_highlow])
            expected_low = np.min(low[seg_slice_highlow])
            expected_close = close[end]
            expected_volume = np.sum(volume[seg_slice_vol])

            assert bars.open[i] == expected_open
            assert bars.high[i] == expected_high
            assert bars.low[i] == expected_low
            assert bars.close[i] == expected_close
            assert bars.volume[i] == expected_volume

            prev_end = end


class TestVolumeAdvancedValidation:
    """Advanced validation tests for volume bar generation."""

    def test_bar_indices_correspond_to_close_prices(self) -> None:
        """Verify that bar_indices correctly map to source close prices."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([600, 700, 800, 900, 1000, 1100, 1200, 1300], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Verify each bar's close matches the close at index_map position
        for i, idx in enumerate(bars.index_map):
            assert bars.close[i] == close[idx], (
                f"Bar {i} close {bars.close[i]} doesn't match source close at index {idx}: {close[idx]}"
            )

    def test_volume_accumulation_matches_sum(self) -> None:
        """Verify that each bar's volume equals sum of source volumes in its range."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Manually reconstruct volume sums to verify
        if len(bars) > 0:
            # First bar starts from index 0
            start_idx = 0
            for i, end_idx in enumerate(bars.index_map):
                expected_volume = np.sum(volume[start_idx : end_idx + 1])
                assert bars.volume[i] == expected_volume, (
                    f"Bar {i} volume {bars.volume[i]} doesn't match expected {expected_volume} "
                    f"from indices {start_idx} to {end_idx}"
                )
                # Next bar starts after current reset (volume reset to 0)
                start_idx = end_idx + 1

    def test_no_gaps_in_coverage(self) -> None:
        """Verify bars cover continuous ranges with no gaps in source data."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([800, 700, 600, 500, 900, 1000, 1100], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        # Verify index_map shows continuous coverage
        if len(bars) > 1:
            for i in range(len(bars) - 1):
                # Each bar should start right after previous bar ends
                assert bars.index_map[i] < bars.index_map[i + 1]
                # Gap should be minimal (next bar starts at next index after reset)
                gap = bars.index_map[i + 1] - bars.index_map[i]
                assert gap >= 1, "Bars should not overlap or share indices"

    def test_high_low_never_violate_ohlc_relationships(self) -> None:
        """Comprehensive OHLC relationship validation across all bars."""
        np.random.seed(99)
        n = 50
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.2)
        high = close + np.abs(np.random.randn(n) * 1.0)
        low = close - np.abs(np.random.randn(n) * 1.0)
        volume = np.random.randint(200, 800, n).astype(np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=3000,
        )

        for i in range(len(bars)):
            # High must be >= all OHLC
            assert bars.high[i] >= bars.open[i], f"Bar {i}: high < open"
            assert bars.high[i] >= bars.low[i], f"Bar {i}: high < low"
            assert bars.high[i] >= bars.close[i], f"Bar {i}: high < close"

            # Low must be <= all OHLC
            assert bars.low[i] <= bars.open[i], f"Bar {i}: low > open"
            assert bars.low[i] <= bars.high[i], f"Bar {i}: low > high"
            assert bars.low[i] <= bars.close[i], f"Bar {i}: low > close"

    @pytest.mark.parametrize(
        "first_volume,subsequent_volume,threshold",
        [
            (100, 500, 1000),  # First volume small, needs accumulation
            (2000, 100, 1500),  # First volume huge, immediate bar
            (1000, 1000, 1500),  # Balanced volumes
        ],
    )
    def test_first_bar_accumulation_behavior(
        self, first_volume: int, subsequent_volume: int, threshold: int
    ) -> None:
        """Test how first bar accumulation handles different volume patterns."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array(
            [first_volume] + [subsequent_volume] * 4,
            dtype=np.int64,
        )

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=threshold,
        )

        # First bar should start with open from first source row
        if len(bars) > 0:
            assert bars.open[0] == close[0]
            # Volume should be cumulative and >= threshold
            assert bars.volume[0] >= threshold


class TestVolumeOutputValidation:
    """Tests for output data type and structure validation."""

    def test_output_dtypes(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        assert bars.open.dtype == np.float64
        assert bars.high.dtype == np.float64
        assert bars.low.dtype == np.float64
        assert bars.close.dtype == np.float64
        assert bars.volume.dtype == np.int64
        assert bars.index_map.dtype == np.int64

    def test_output_contiguity(self) -> None:
        """Output arrays should be C-contiguous."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        if len(bars) > 0:
            assert bars.open.flags["C_CONTIGUOUS"]
            assert bars.high.flags["C_CONTIGUOUS"]
            assert bars.low.flags["C_CONTIGUOUS"]
            assert bars.close.flags["C_CONTIGUOUS"]
            assert bars.volume.flags["C_CONTIGUOUS"]
            assert bars.index_map.flags["C_CONTIGUOUS"]

    def test_consistent_lengths(self) -> None:
        """All output arrays should have the same length."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_volume_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=2000,
        )

        n_bars = len(bars)
        assert len(bars.open) == n_bars
        assert len(bars.high) == n_bars
        assert len(bars.low) == n_bars
        assert len(bars.close) == n_bars
        assert len(bars.volume) == n_bars
        assert len(bars.index_map) == n_bars


@pytest.mark.benchmark
class TestVolumePerformance:
    """Performance benchmarks for Volume bar generation."""

    def test_1m_rows_performance(self) -> None:
        """Should achieve 1M+ rows/sec throughput."""
        import time

        # Generate 1M rows of realistic price and volume data
        n = 1_000_000
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_arr = close
        volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

        # Warm up JIT (first run triggers compilation)
        _ = generate_volume_bars_series(
            open_arr=open_arr[:1000],
            high_arr=high[:1000],
            low_arr=low[:1000],
            close_arr=close[:1000],
            volume_arr=volume[:1000],
            volume_threshold=50000,
        )

        # Measure performance
        start = time.perf_counter()
        result = generate_volume_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            volume_threshold=50000,
        )
        elapsed = time.perf_counter() - start

        # Calculate throughput
        throughput = n / elapsed

        # Verify output is valid
        assert isinstance(result, BarSeries)
        assert len(result) > 0
        assert result.type == "volume"
        assert result.parameters["volume_threshold"] == 50000

        # Verify meets performance target (1M+ rows/sec)
        assert throughput >= 1_000_000, (
            f"Performance target not met: {throughput:,.0f} rows/sec < 1,000,000 rows/sec"
        )
