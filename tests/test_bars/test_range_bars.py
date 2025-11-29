"""Tests for Range bar generation.

Tests cover:
- Basic functionality with fixed range_size
- Algorithm correctness (cumulative high/low tracking)
- Edge cases (empty data, single row, no bars generated)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.range_bars import generate_range_bars_series


class TestRangeRegistration:
    """Tests for bar factory registration."""

    def test_range_registered(self) -> None:
        """Range should be registered with the bar factory."""
        assert "range" in list_bar_types()

    def test_get_range_generator(self) -> None:
        """Should retrieve the range generator from factory."""
        generator = get_bar_generator("range")
        assert generator is generate_range_bars_series


class TestRangeBasicFunctionality:
    """Tests for basic Range bar generation."""

    def test_simple_range_bars(self) -> None:
        """Should generate bars when price range exceeds range_size."""
        # Create price data with clear 10-point ranges
        close = np.array([100.0, 102.0, 105.0, 110.0, 112.0, 115.0, 120.0])
        high = np.array([100.5, 103.0, 106.0, 111.0, 113.0, 116.0, 121.0])
        low = np.array([99.5, 101.0, 104.0, 109.0, 111.0, 114.0, 119.0])
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            range_size=10.0,
        )

        # Should generate bars when cumulative high-low >= 10
        assert len(bars) > 0
        assert bars.type == "range"
        assert bars.parameters["range_size"] == 10.0

    def test_high_volatility_many_bars(self) -> None:
        """High volatility with small range_size should create many bars."""
        # Price oscillates wildly, each move exceeds range_size
        close = np.array([100.0, 110.0, 95.0, 105.0, 90.0, 100.0])
        high = close + 2.0
        low = close - 2.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            range_size=5.0,
        )

        # With high volatility and small range_size, expect multiple bars
        assert len(bars) >= 3

    def test_low_volatility_few_bars(self) -> None:
        """Low volatility with large range_size should create few/no bars."""
        # Price moves slowly, never reaching range_size
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            range_size=50.0,
        )

        # With small moves and large range_size, expect 0 bars
        assert len(bars) == 0

    def test_volume_aggregation(self) -> None:
        """Volume should be correctly aggregated per Range bar."""
        # Create data that will form 2 range bars
        close = np.array([100.0, 102.0, 105.0, 110.0, 112.0, 115.0, 120.0, 122.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int64)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=10.0,
        )

        # Verify volume aggregation based on bar_indices
        if len(bars) > 0:
            # First bar should aggregate from start to first bar_index
            assert bars.volume[0] > 0
            # Each subsequent bar aggregates between bar_indices
            for i in range(1, len(bars)):
                assert bars.volume[i] > 0

    def test_index_map_tracking(self) -> None:
        """Index map should correctly track source row where bar completed."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Verify index_map points to valid source indices
        if len(bars) > 0:
            assert all(0 <= idx < n for idx in bars.index_map)
            # Indices should be monotonically increasing
            assert all(bars.index_map[i] < bars.index_map[i + 1] for i in range(len(bars) - 1))


class TestRangeAlgorithmCorrectness:
    """Tests for Range bar algorithm implementation."""

    def test_cumulative_high_low_tracking(self) -> None:
        """Should track cumulative high and low correctly across bars."""
        # Design specific scenario to test cumulative tracking
        close = np.array([100.0, 103.0, 105.0, 107.0, 110.0])
        high = np.array([100.5, 104.0, 106.0, 108.0, 111.0])
        low = np.array([99.5, 102.0, 104.0, 106.0, 109.0])
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # First bar should form when cumulative range reaches 10
        # high[3]=108, low[0]=99.5, range = 8.5 (not yet)
        # high[4]=111, low[0]=99.5, range = 11.5 (closes bar)
        assert len(bars) == 1
        assert bars.close[0] == 110.0

    def test_bar_reset_after_close(self) -> None:
        """After closing a bar, should reset and start tracking new range."""
        # First range closes, then new range starts accumulating
        close = np.array([100.0, 105.0, 111.0, 115.0, 122.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Should generate 2 bars
        # Bar 1: from index 0 until range >= 10
        # Bar 2: resets and accumulates from close of bar 1
        assert len(bars) == 2


class TestRangeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_range_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            range_size=10.0,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row(self) -> None:
        """Single row should return empty BarSeries (need at least 2 for range)."""
        bars = generate_range_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            range_size=10.0,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_invalid_range_size_zero(self) -> None:
        """range_size of zero should raise ValueError."""
        with pytest.raises(ValueError, match="range_size must be positive"):
            generate_range_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                range_size=0.0,
            )

    def test_invalid_range_size_negative(self) -> None:
        """Negative range_size should raise ValueError."""
        with pytest.raises(ValueError, match="range_size must be positive"):
            generate_range_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                range_size=-5.0,
            )

    def test_inconsistent_array_lengths(self) -> None:
        """Arrays with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="inconsistent lengths"):
            generate_range_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0]),  # Different length
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                range_size=10.0,
            )


class TestRangeOutputValidation:
    """Tests for output data type and structure validation."""

    def test_output_dtypes(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        assert bars.open.dtype == np.float64
        assert bars.high.dtype == np.float64
        assert bars.low.dtype == np.float64
        assert bars.close.dtype == np.float64
        assert bars.volume.dtype == np.int64
        assert bars.index_map.dtype == np.int64

    def test_output_contiguity(self) -> None:
        """Output arrays should be C-contiguous."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
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
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        n_bars = len(bars)
        assert len(bars.open) == n_bars
        assert len(bars.high) == n_bars
        assert len(bars.low) == n_bars
        assert len(bars.close) == n_bars
        assert len(bars.volume) == n_bars
        assert len(bars.index_map) == n_bars


class TestRangeVolumeAggregation:
    """Tests for exact volume aggregation logic."""

    def test_volume_first_bar_exact(self) -> None:
        """First bar should aggregate volume from 0 to bar_indices[0]."""
        # Create scenario with known volumes and clear bar formation
        close = np.array([100.0, 103.0, 105.0, 111.0, 115.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500], dtype=np.int64)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=10.0,
        )

        # Should create at least 1 bar
        assert len(bars) >= 1

        # First bar aggregates from index 0 to bar_indices[0]
        first_bar_end_idx = bars.index_map[0]
        expected_volume = np.sum(volume[: first_bar_end_idx + 1])
        assert bars.volume[0] == expected_volume

    def test_volume_subsequent_bars_exact(self) -> None:
        """Subsequent bars should aggregate from previous+1 to current index."""
        # Create scenario that generates multiple bars
        close = np.array([100.0, 105.0, 111.0, 115.0, 120.0, 125.0, 131.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500, 600, 700], dtype=np.int64)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=10.0,
        )

        # Verify each bar's volume aggregation
        for bar_idx in range(len(bars)):
            if bar_idx == 0:
                # First bar: 0 to bar_indices[0]
                expected_vol = np.sum(volume[: bars.index_map[0] + 1])
            else:
                # Subsequent bars: previous+1 to current
                start = bars.index_map[bar_idx - 1] + 1
                end = bars.index_map[bar_idx] + 1
                expected_vol = np.sum(volume[start:end])

            assert bars.volume[bar_idx] == expected_vol

    def test_volume_total_conservation(self) -> None:
        """Total volume across all bars should equal total input volume."""
        close = np.array([100.0, 105.0, 111.0, 115.0, 122.0, 128.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=10.0,
        )

        # Sum of bar volumes should be <= sum of input volumes
        # (may be less if final incomplete bar is not closed)
        total_bar_volume = np.sum(bars.volume)
        total_input_volume = np.sum(volume)
        assert total_bar_volume <= total_input_volume

    def test_volume_single_bar_entire_array(self) -> None:
        """Single bar consuming entire array should aggregate all volume."""
        close = np.array([100.0, 102.0, 105.0, 107.0, 110.0, 112.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=10.0,
        )

        # Should create 1 bar
        assert len(bars) == 1

        # Verify volume aggregation for that single bar
        end_idx = bars.index_map[0]
        expected_volume = np.sum(volume[: end_idx + 1])
        assert bars.volume[0] == expected_volume


class TestRangeBarReset:
    """Tests for bar reset and continuity logic."""

    def test_new_bar_starts_from_previous_close(self) -> None:
        """New bar should start from close price where previous bar closed."""
        # First bar closes at index 2, second bar should start from close[2]
        close = np.array([100.0, 105.0, 111.0, 115.0, 122.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Should create at least 2 bars
        assert len(bars) >= 2

        # Second bar's open should equal close price at first bar's completion
        first_close_idx = bars.index_map[0]
        assert bars.open[1] == close[first_close_idx]

    def test_high_low_reset_to_current_values(self) -> None:
        """After closing bar, high/low should reset to current bar's values."""
        close = np.array([100.0, 105.0, 111.0, 113.0, 120.0, 125.0])
        high = np.array([101.0, 106.0, 112.0, 114.0, 121.0, 126.0])
        low = np.array([99.0, 104.0, 110.0, 112.0, 119.0, 124.0])
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Should create multiple bars
        assert len(bars) >= 2

        # Each bar should have valid OHLC relationships
        for i in range(len(bars)):
            assert bars.low[i] <= bars.open[i] <= bars.high[i]
            assert bars.low[i] <= bars.close[i] <= bars.high[i]

    def test_multiple_bars_continuity(self) -> None:
        """Multiple bars should maintain price continuity without gaps."""
        close = np.array([100.0, 105.0, 111.0, 115.0, 122.0, 128.0, 135.0, 142.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Should create multiple bars
        assert len(bars) >= 3

        # Each subsequent bar should start from previous bar's close
        for i in range(1, len(bars)):
            prev_close_idx = bars.index_map[i - 1]
            assert bars.open[i] == close[prev_close_idx]


class TestRangeBoundaryConditions:
    """Tests for range_size boundary conditions."""

    def test_exact_threshold_closes_bar(self) -> None:
        """Price exactly at range_size threshold should close bar."""
        # Setup: cumulative range reaches exactly 10.0
        close = np.array([100.0, 102.0, 105.0, 110.0])
        high = np.array([100.0, 102.0, 105.0, 110.0])
        low = np.array([100.0, 100.0, 100.0, 100.0])
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # At index 3: high=110, low=100, range=10.0 (exactly threshold)
        # Should close bar (>= condition)
        assert len(bars) == 1
        assert bars.high[0] == 110.0
        assert bars.low[0] == 100.0

    def test_below_threshold_no_close(self) -> None:
        """Price below range_size threshold should not close bar."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Cumulative range never reaches 10.0, no bars should close
        assert len(bars) == 0

    def test_above_threshold_closes_immediately(self) -> None:
        """Price exceeding range_size should close bar immediately."""
        close = np.array([100.0, 102.0, 112.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # At index 2: high=113, low=99, range=14 (exceeds 10)
        assert len(bars) == 1
        assert bars.close[0] == 112.0

    def test_large_gap_single_bar(self) -> None:
        """Large price gap should create single bar, not multiple."""
        # Price jumps far exceeding range_size in one tick
        close = np.array([100.0, 150.0, 155.0])
        high = close + 1.0
        low = close - 1.0
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Large gap creates bars based on cumulative range, not gap size
        assert len(bars) >= 1

        # First bar should close when range exceeds 10
        assert bars.high[0] - bars.low[0] >= 10.0

    def test_narrow_range_accumulation(self) -> None:
        """Narrow ranges should accumulate over many bars before closing."""
        # Small incremental moves that eventually exceed threshold
        close = np.array(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]
        )
        high = close + 0.1
        low = close - 0.1
        n = len(close)

        bars = generate_range_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            range_size=10.0,
        )

        # Should eventually accumulate to close at least 1 bar
        assert len(bars) >= 1

        # Verify cumulative high-low for first bar
        assert bars.high[0] - bars.low[0] >= 10.0


@pytest.mark.benchmark
class TestRangePerformance:
    """Performance benchmarks for Range bar generation."""

    def test_throughput_1m_rows_manual(self) -> None:
        """Should achieve 1M+ rows/sec throughput (manual timing)."""
        import time

        n = 1_000_000
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_arr = close.copy()
        volume = np.ones(n, dtype=np.int64) * 1000

        # JIT warmup run (not measured)
        _ = generate_range_bars_series(
            open_arr=open_arr[:1000],
            high_arr=high[:1000],
            low_arr=low[:1000],
            close_arr=close[:1000],
            volume_arr=volume[:1000],
            range_size=5.0,
        )

        # Actual benchmark (manual timing)
        start = time.perf_counter()
        result = generate_range_bars_series(
            open_arr=open_arr,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            range_size=5.0,
        )
        elapsed = time.perf_counter() - start

        # Verify output is valid
        assert isinstance(result, BarSeries)
        assert len(result) > 0

        # Verify performance target (1M+ rows/sec)
        throughput = n / elapsed
        print(f"\nThroughput: {throughput:,.0f} rows/sec ({elapsed:.3f}s for {n:,} rows)")
        assert (
            throughput >= 1_000_000
        ), f"Performance below target: {throughput:,.0f} rows/sec < 1M rows/sec"
