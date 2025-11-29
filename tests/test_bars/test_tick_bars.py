"""Tests for Tick bar generation.

Tests cover:
- Basic functionality with fixed tick_threshold
- Algorithm correctness (N bars per output bar)
- Edge cases (empty data, partial final bar, tick_threshold > n)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.tick_bars import generate_tick_bars_series


class TestTickRegistration:
    """Tests for bar factory registration."""

    def test_tick_registered(self) -> None:
        """Tick should be registered with the bar factory."""
        assert "tick" in list_bar_types()

    def test_get_tick_generator(self) -> None:
        """Should retrieve the tick generator from factory."""
        generator = get_bar_generator("tick")
        assert generator is generate_tick_bars_series


class TestTickBasicFunctionality:
    """Tests for basic Tick bar generation."""

    def test_exact_division(self) -> None:
        """tick_threshold that divides evenly should create exact bars."""
        # 10 source bars with tick_threshold=5 should create 2 tick bars
        close = np.arange(100.0, 110.0)  # 10 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=5,
        )

        assert len(bars) == 2
        assert bars.type == "tick"
        assert bars.parameters["tick_threshold"] == 5

        # First bar: aggregates source bars 0-4
        assert bars.open[0] == 100.0
        assert bars.close[0] == 104.0
        assert bars.high[0] == 105.0  # max(high[0:5]) = 105.0
        assert bars.low[0] == 99.0  # min(low[0:5]) = 99.0

        # Second bar: aggregates source bars 5-9
        assert bars.open[1] == 105.0
        assert bars.close[1] == 109.0
        assert bars.high[1] == 110.0
        assert bars.low[1] == 104.0

    def test_partial_final_bar(self) -> None:
        """Final bar with fewer than tick_threshold bars should still be created."""
        # 7 source bars with tick_threshold=5 should create 2 bars (5 + 2)
        close = np.arange(100.0, 107.0)  # 7 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=5,
        )

        assert len(bars) == 2

        # First bar: 5 source bars
        assert bars.open[0] == 100.0
        assert bars.close[0] == 104.0

        # Second bar: 2 source bars (partial)
        assert bars.open[1] == 105.0
        assert bars.close[1] == 106.0

    def test_tick_threshold_equals_n(self) -> None:
        """tick_threshold equal to n should create single bar."""
        close = np.arange(100.0, 105.0)  # 5 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=5,
        )

        assert len(bars) == 1
        assert bars.open[0] == 100.0
        assert bars.close[0] == 104.0

    def test_tick_threshold_greater_than_n(self) -> None:
        """tick_threshold > n should create single bar with all data."""
        close = np.arange(100.0, 105.0)  # 5 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=10,  # Greater than n=5
        )

        assert len(bars) == 1
        assert bars.open[0] == 100.0
        assert bars.close[0] == 104.0
        assert bars.high[0] == 105.0
        assert bars.low[0] == 99.0

    def test_volume_aggregation(self) -> None:
        """Volume should be correctly aggregated per Tick bar."""
        close = np.arange(100.0, 110.0)  # 10 bars
        high = close + 1.0
        low = close - 1.0
        volume = np.arange(100, 1100, 100, dtype=np.int64)  # 100, 200, ..., 1000

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            tick_threshold=5,
        )

        assert len(bars) == 2

        # First bar: sum of volumes for indices 0-4 = 100+200+300+400+500 = 1500
        assert bars.volume[0] == 1500

        # Second bar: sum of volumes for indices 5-9 = 600+700+800+900+1000 = 4000
        assert bars.volume[1] == 4000

    def test_index_map_tracking(self) -> None:
        """Index map should correctly track source row where bar completed."""
        close = np.arange(100.0, 110.0)  # 10 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=5,
        )

        # First bar completes at index 4, second at index 9
        np.testing.assert_array_equal(bars.index_map, [4, 9])


class TestTickAlgorithmCorrectness:
    """Tests for Tick bar algorithm implementation."""

    def test_ohlc_aggregation_correctness(self) -> None:
        """OHLC values should be correctly aggregated from source bars."""
        # Design specific OHLC pattern to test aggregation
        open_arr = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = np.array([102.0, 103.0, 105.0, 104.0, 106.0, 107.0])
        low = np.array([99.0, 100.0, 101.0, 102.0, 103.0, 104.0])
        close = np.array([101.0, 102.0, 104.0, 103.0, 105.0, 106.0])
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=3,
        )

        assert len(bars) == 2

        # First bar (indices 0-2)
        assert bars.open[0] == 100.0  # open[0]
        assert bars.close[0] == 104.0  # close[2]
        assert bars.high[0] == 105.0  # max(high[0:3]) = max(102, 103, 105)
        assert bars.low[0] == 99.0  # min(low[0:3]) = min(99, 100, 101)

        # Second bar (indices 3-5)
        assert bars.open[1] == 103.0  # open[3]
        assert bars.close[1] == 106.0  # close[5]
        assert bars.high[1] == 107.0  # max(high[3:6]) = max(104, 106, 107)
        assert bars.low[1] == 102.0  # min(low[3:6]) = min(102, 103, 104)

    def test_tick_threshold_one(self) -> None:
        """tick_threshold=1 should create 1:1 mapping (no aggregation)."""
        close = np.array([100.0, 101.0, 102.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=1,
        )

        # Should create 3 bars, each with one source bar
        assert len(bars) == 3
        np.testing.assert_array_almost_equal(bars.open, close)
        np.testing.assert_array_almost_equal(bars.close, close)
        np.testing.assert_array_almost_equal(bars.high, high)
        np.testing.assert_array_almost_equal(bars.low, low)

    def test_ohlc_aggregation_with_tick_threshold_2(self) -> None:
        """OHLC aggregation with tick_threshold=2 should correctly pair bars."""
        # Design specific pattern
        open_arr = np.array([100.0, 101.0, 102.0, 103.0])
        high = np.array([102.0, 103.0, 105.0, 107.0])
        low = np.array([99.0, 100.0, 101.0, 102.0])
        close = np.array([101.0, 102.0, 104.0, 106.0])
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=2,
        )

        assert len(bars) == 2

        # First bar (indices 0-1)
        assert bars.open[0] == 100.0  # open[0]
        assert bars.close[0] == 102.0  # close[1]
        assert bars.high[0] == 103.0  # max(102, 103)
        assert bars.low[0] == 99.0  # min(99, 100)

        # Second bar (indices 2-3)
        assert bars.open[1] == 102.0  # open[2]
        assert bars.close[1] == 106.0  # close[3]
        assert bars.high[1] == 107.0  # max(105, 107)
        assert bars.low[1] == 101.0  # min(101, 102)

    def test_ohlc_aggregation_with_tick_threshold_5(self) -> None:
        """OHLC aggregation with tick_threshold=5 should handle larger groups."""
        # Create 10 bars that will be split into 2 groups of 5
        open_arr = np.array([100.0, 101.0, 102.0, 103.0, 104.0,
                            105.0, 106.0, 107.0, 108.0, 109.0])
        high = np.array([102.0, 103.0, 108.0, 104.0, 106.0,
                        107.0, 115.0, 109.0, 110.0, 111.0])
        low = np.array([99.0, 100.0, 101.0, 102.0, 103.0,
                       104.0, 105.0, 106.0, 107.0, 108.0])
        close = np.array([101.0, 102.0, 104.0, 103.0, 105.0,
                         106.0, 107.0, 108.0, 109.0, 110.0])
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=5,
        )

        assert len(bars) == 2

        # First bar (indices 0-4)
        assert bars.open[0] == 100.0  # open[0]
        assert bars.close[0] == 105.0  # close[4]
        assert bars.high[0] == 108.0  # max of first 5 high values
        assert bars.low[0] == 99.0  # min of first 5 low values

        # Second bar (indices 5-9)
        assert bars.open[1] == 105.0  # open[5]
        assert bars.close[1] == 110.0  # close[9]
        assert bars.high[1] == 115.0  # max of second 5 high values
        assert bars.low[1] == 104.0  # min of second 5 low values

    def test_bar_indices_monotonicity(self) -> None:
        """Bar indices should be strictly monotonically increasing."""
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(100) * 2)
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=close + 1,
            low_arr=close - 1,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=5,
        )

        if len(bars) > 1:
            diffs = np.diff(bars.index_map)
            assert np.all(diffs > 0), (
                "bar_indices should be strictly monotonically increasing"
            )

    def test_bar_indices_correctness(self) -> None:
        """Bar indices should point to correct source bar completion index."""
        close = np.arange(100.0, 110.0)  # 10 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=3,
        )

        # Should create 4 bars: [0-2], [3-5], [6-8], [9]
        assert len(bars) == 4
        assert bars.index_map[0] == 2  # First bar completes at index 2
        assert bars.index_map[1] == 5  # Second bar completes at index 5
        assert bars.index_map[2] == 8  # Third bar completes at index 8
        assert bars.index_map[3] == 9  # Fourth bar completes at index 9 (partial)


class TestTickEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_tick_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            tick_threshold=5,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row(self) -> None:
        """Single row should create single tick bar."""
        bars = generate_tick_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            tick_threshold=5,
        )

        assert len(bars) == 1
        assert bars.open[0] == 100.0
        assert bars.close[0] == 100.0

    def test_invalid_tick_threshold_zero(self) -> None:
        """tick_threshold of zero should raise ValueError."""
        with pytest.raises(ValueError, match="tick_threshold must be positive"):
            generate_tick_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                tick_threshold=0,
            )

    def test_invalid_tick_threshold_negative(self) -> None:
        """Negative tick_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="tick_threshold must be positive"):
            generate_tick_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                tick_threshold=-5,
            )

    def test_inconsistent_array_lengths(self) -> None:
        """Arrays with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="inconsistent lengths"):
            generate_tick_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0]),  # Different length
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                tick_threshold=5,
            )

    def test_very_large_tick_threshold(self) -> None:
        """tick_threshold much larger than n should create single bar."""
        close = np.arange(100.0, 105.0)  # 5 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        # tick_threshold = 10x data length
        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            tick_threshold=50,
        )

        # Should still create 1 bar containing all data
        assert len(bars) == 1
        assert bars.open[0] == 100.0
        assert bars.close[0] == 104.0
        assert bars.high[0] == 105.0
        assert bars.low[0] == 99.0
        assert bars.volume[0] == 500  # Sum of 5 * 100

    def test_varying_volume_patterns(self) -> None:
        """Volume aggregation should work correctly with varying patterns."""
        close = np.arange(100.0, 106.0)  # 6 bars
        high = close + 1.0
        low = close - 1.0

        # Increasing volume pattern
        volume = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            tick_threshold=2,
        )

        assert len(bars) == 3
        # First bar: 100 + 200 = 300
        assert bars.volume[0] == 300
        # Second bar: 300 + 400 = 700
        assert bars.volume[1] == 700
        # Third bar: 500 + 600 = 1100
        assert bars.volume[2] == 1100

    def test_zero_volume_pattern(self) -> None:
        """Should handle all zero volumes correctly."""
        close = np.arange(100.0, 105.0)  # 5 bars
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.zeros(n, dtype=np.int64),
            tick_threshold=2,
        )

        # Should create 3 bars: [0-1], [2-3], [4]
        assert len(bars) == 3
        # All volumes should be 0
        assert np.all(bars.volume == 0)


class TestTickOutputValidation:
    """Tests for output data type and structure validation."""

    def test_output_dtypes(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.arange(100.0, 110.0)
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=5,
        )

        assert bars.open.dtype == np.float64
        assert bars.high.dtype == np.float64
        assert bars.low.dtype == np.float64
        assert bars.close.dtype == np.float64
        assert bars.volume.dtype == np.int64
        assert bars.index_map.dtype == np.int64

    def test_output_contiguity(self) -> None:
        """Output arrays should be C-contiguous."""
        close = np.arange(100.0, 110.0)
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=5,
        )

        assert bars.open.flags["C_CONTIGUOUS"]
        assert bars.high.flags["C_CONTIGUOUS"]
        assert bars.low.flags["C_CONTIGUOUS"]
        assert bars.close.flags["C_CONTIGUOUS"]
        assert bars.volume.flags["C_CONTIGUOUS"]
        assert bars.index_map.flags["C_CONTIGUOUS"]

    def test_consistent_lengths(self) -> None:
        """All output arrays should have the same length."""
        close = np.arange(100.0, 110.0)
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_tick_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            tick_threshold=5,
        )

        n_bars = len(bars)
        assert len(bars.open) == n_bars
        assert len(bars.high) == n_bars
        assert len(bars.low) == n_bars
        assert len(bars.close) == n_bars
        assert len(bars.volume) == n_bars
        assert len(bars.index_map) == n_bars


@pytest.mark.benchmark
class TestTickPerformance:
    """Performance benchmarks for Tick bar generation."""

    def test_1m_rows_performance(self) -> None:
        """Should achieve 1M+ rows/sec throughput."""
        import time

        # Generate 1M rows of realistic price data
        n = 1_000_000
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_arr = close
        volume = np.ones(n, dtype=np.int64) * 1000

        # Warm up JIT (first run triggers compilation)
        _ = generate_tick_bars_series(
            open_arr=open_arr[:1000],
            high_arr=high[:1000],
            low_arr=low[:1000],
            close_arr=close[:1000],
            volume_arr=volume[:1000],
            tick_threshold=10,
        )

        # Measure performance
        start = time.perf_counter()
        result = generate_tick_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            tick_threshold=10,
        )
        elapsed = time.perf_counter() - start

        # Calculate throughput
        throughput = n / elapsed

        # Verify output is valid
        assert isinstance(result, BarSeries)
        assert len(result) > 0
        # Should create exactly 100,000 bars (1M / 10)
        assert len(result) == 100_000

        # Verify meets performance target
        assert throughput >= 1_000_000, (
            f"Throughput {throughput:,.0f} rows/sec < 1M rows/sec target"
        )
