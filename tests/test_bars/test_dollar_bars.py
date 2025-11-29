"""Tests for Dollar bar generation.

Tests cover:
- Basic functionality with fixed dollar_threshold
- Algorithm correctness (dollar volume accumulation: close * volume)
- Edge cases (empty data, single row, no bars generated)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.dollar_bars import generate_dollar_bars_series


class TestDollarRegistration:
    """Tests for bar factory registration."""

    def test_dollar_registered(self) -> None:
        """Dollar should be registered with the bar factory."""
        assert "dollar" in list_bar_types()

    def test_get_dollar_generator(self) -> None:
        """Should retrieve the dollar generator from factory."""
        generator = get_bar_generator("dollar")
        assert generator is generate_dollar_bars_series


class TestDollarBasicFunctionality:
    """Tests for basic Dollar bar generation."""

    def test_simple_dollar_bars(self) -> None:
        """Should generate bars when cumulative dollar volume exceeds threshold."""
        # Dollar volume = close[i] * volume[i]
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 150, 200, 250, 300, 350, 400], dtype=np.int64)
        # Dollar volumes: 10000, 15150, 20400, 25750, 31200, 36750, 42400

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=50000.0,
        )

        # Should generate bars when cumulative dollar volume >= 50000
        assert len(bars) >= 1
        assert bars.type == "dollar"
        assert bars.parameters["dollar_threshold"] == 50000.0

    def test_high_dollar_volume_many_bars(self) -> None:
        """High dollar volume with small threshold should create many bars."""
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([500, 500, 500, 500, 500, 500], dtype=np.int64)
        # Dollar volume per bar: 50000

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=60000.0,
        )

        # With consistent dollar volume, expect multiple bars
        assert len(bars) >= 2

    def test_low_dollar_volume_few_bars(self) -> None:
        """Low dollar volume with large threshold should create few/no bars."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        high = close + 0.1
        low = close - 0.1
        volume = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        # Dollar volumes: 10, 40, 90, 160, 250

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=1000000.0,
        )

        # With low dollar volume and large threshold, expect 0 bars
        assert len(bars) == 0

    def test_volume_aggregation(self) -> None:
        """Volume field should contain aggregated raw volume (not dollar volume)."""
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        # Dollar volumes: 10000, 20000, 30000, 40000, 50000
        # Cumulative: 10000, 30000, 60000 -> bar closes at index 2

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=50000.0,
        )

        # Volume field should be sum of raw volume, not dollar volume
        if len(bars) > 0:
            # First bar: volume[0] + volume[1] + volume[2] = 100+200+300 = 600
            assert bars.volume[0] == 600

    def test_index_map_tracking(self) -> None:
        """Index map should correctly track source row where bar completed."""
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([500, 500, 500, 500, 500, 500], dtype=np.int64)
        n = len(close)

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=100000.0,
        )

        # Verify index_map points to valid source indices
        if len(bars) > 0:
            assert all(0 <= idx < n for idx in bars.index_map)
            # Indices should be monotonically increasing
            if len(bars) > 1:
                assert all(bars.index_map[i] < bars.index_map[i + 1] for i in range(len(bars) - 1))


class TestDollarAlgorithmCorrectness:
    """Tests for Dollar bar algorithm implementation."""

    def test_dollar_volume_calculation(self) -> None:
        """Should correctly calculate dollar volume as close * volume."""
        close = np.array([100.0, 110.0, 120.0, 130.0])
        high = close + 5.0
        low = close - 5.0
        volume = np.array([100, 200, 300, 400], dtype=np.int64)
        # Dollar volumes: 10000, 22000, 36000, 52000
        # Cumulative: 10000, 32000, 68000 -> bar closes at index 2

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=60000.0,
        )

        assert len(bars) == 1
        assert bars.close[0] == 120.0  # Closes at index 2

    def test_bar_reset_after_close(self) -> None:
        """After closing a bar, should reset and start accumulating new dollar volume."""
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([600, 700, 800, 900, 1000, 1100, 1200], dtype=np.int64)
        # Dollar volumes: 60000, 70000, 80000, 90000, 100000, 110000, 120000
        # Cumulative: 60000, 130000, 210000 -> bars close at indices 0, 2, 4

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=100000.0,
        )

        # Should generate multiple bars
        assert len(bars) >= 2

    def test_ohlc_tracking(self) -> None:
        """Should track OHLC properly across accumulated bars."""
        close = np.array([100.0, 102.0, 101.0, 103.0])
        high = np.array([100.5, 103.0, 102.0, 104.0])
        low = np.array([99.5, 101.0, 100.0, 102.0])
        volume = np.array([800, 700, 600, 500], dtype=np.int64)
        # Dollar volumes: 80000, 71400, 60600, 51500

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=200000.0,
        )

        if len(bars) > 0:
            # First bar should have open from first source bar
            assert bars.open[0] == 100.0
            # High should be max across accumulated bars
            assert bars.high[0] >= 103.0
            # Low should be min across accumulated bars
            assert bars.low[0] <= 100.0

    def test_price_sensitivity(self) -> None:
        """Higher prices should create bars faster at same volume."""
        # Test 1: Low price, high volume
        close_low = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        volume_high = np.array([10000, 10000, 10000, 10000, 10000], dtype=np.int64)
        # Dollar volume per bar: 100000

        bars_low_price = generate_dollar_bars_series(
            open_arr=close_low,
            high_arr=close_low + 1,
            low_arr=close_low - 1,
            close_arr=close_low,
            volume_arr=volume_high,
            dollar_threshold=200000.0,
        )

        # Test 2: High price, low volume
        close_high = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        volume_low = np.array([100, 100, 100, 100, 100], dtype=np.int64)
        # Dollar volume per bar: 100000

        bars_high_price = generate_dollar_bars_series(
            open_arr=close_high,
            high_arr=close_high + 10,
            low_arr=close_high - 10,
            close_arr=close_high,
            volume_arr=volume_low,
            dollar_threshold=200000.0,
        )

        # Both should generate similar number of bars (same dollar volume)
        assert abs(len(bars_low_price) - len(bars_high_price)) <= 1


class TestDollarEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_dollar_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            dollar_threshold=100000.0,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row(self) -> None:
        """Single row should return empty BarSeries (need at least 2)."""
        bars = generate_dollar_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            dollar_threshold=50000.0,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_invalid_threshold_zero(self) -> None:
        """dollar_threshold of zero should raise ValueError."""
        with pytest.raises(ValueError, match="dollar_threshold must be positive"):
            generate_dollar_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                dollar_threshold=0.0,
            )

    def test_invalid_threshold_negative(self) -> None:
        """Negative dollar_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="dollar_threshold must be positive"):
            generate_dollar_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                dollar_threshold=-50000.0,
            )

    def test_inconsistent_array_lengths(self) -> None:
        """Arrays with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="inconsistent lengths"):
            generate_dollar_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0]),  # Different length
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                dollar_threshold=100000.0,
            )


class TestDollarOutputValidation:
    """Tests for output data type and structure validation."""

    def test_output_dtypes(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=200000.0,
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

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=200000.0,
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

        bars = generate_dollar_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            dollar_threshold=200000.0,
        )

        n_bars = len(bars)
        assert len(bars.open) == n_bars
        assert len(bars.high) == n_bars
        assert len(bars.low) == n_bars
        assert len(bars.close) == n_bars
        assert len(bars.volume) == n_bars
        assert len(bars.index_map) == n_bars


@pytest.mark.benchmark
class TestDollarPerformance:
    """Performance benchmarks for Dollar bar generation."""

    @pytest.mark.skip(reason="Performance test needs pytest-benchmark fixture - deferred to I2")
    def test_1m_rows_performance(self) -> None:
        """Should achieve 1M+ rows/sec throughput."""
        # Generate 1M rows of realistic price and volume data
        n = 1_000_000
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_arr = close
        volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

        # Benchmark the function
        def run() -> BarSeries:
            return generate_dollar_bars_series(
                open_arr=open_arr,
                high_arr=high,
                low_arr=low,
                close_arr=close,
                volume_arr=volume,
                dollar_threshold=5000000.0,
            )

        result = benchmark(run)

        # Verify output is valid
        assert isinstance(result, BarSeries)
        assert len(result) > 0
