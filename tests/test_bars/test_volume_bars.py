"""Tests for Volume bar generation.

Tests cover:
- Basic functionality with fixed volume_threshold
- Algorithm correctness (volume accumulation tracking)
- Edge cases (empty data, single row, no bars generated)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

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
        n = len(close)

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
        n = len(close)

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
            assert all(0 <= idx < n for idx in bars.index_map)
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
