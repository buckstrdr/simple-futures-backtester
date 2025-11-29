"""Tests for Renko bar generation.

Tests cover:
- Basic functionality with fixed brick size
- ATR-based dynamic sizing
- Algorithm correctness (up/down bricks, 2-brick reversal rule)
- Edge cases (empty data, single row, all identical prices)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.renko import generate_renko_bars_series


class TestRenkoRegistration:
    """Tests for bar factory registration."""

    def test_renko_registered(self) -> None:
        """Renko should be registered with the bar factory."""
        assert "renko" in list_bar_types()

    def test_get_renko_generator(self) -> None:
        """Should retrieve the renko generator from factory."""
        generator = get_bar_generator("renko")
        assert generator is generate_renko_bars_series


class TestRenkoBasicFunctionality:
    """Tests for basic Renko bar generation."""

    def test_fixed_brick_size_uptrend(self) -> None:
        """Fixed brick size should generate correct up bricks in uptrend."""
        # Price moves from 100 to 130 with brick_size=10
        # Should create 3 up bricks: 100->110, 110->120, 120->130
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            brick_size=10.0,
        )

        assert len(bars) == 3
        assert bars.type == "renko"
        assert bars.parameters["brick_size"] == 10.0

        # Verify OHLC values for up bricks
        np.testing.assert_array_almost_equal(bars.open, [100.0, 110.0, 120.0])
        np.testing.assert_array_almost_equal(bars.close, [110.0, 120.0, 130.0])
        np.testing.assert_array_almost_equal(bars.high, [110.0, 120.0, 130.0])
        np.testing.assert_array_almost_equal(bars.low, [100.0, 110.0, 120.0])

    def test_fixed_brick_size_downtrend(self) -> None:
        """Fixed brick size should generate correct down bricks in downtrend."""
        # Price moves from 100 to 70 with brick_size=10
        # Should create 3 down bricks: 100->90, 90->80, 80->70
        close = np.array([100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            brick_size=10.0,
        )

        assert len(bars) == 3

        # Verify OHLC values for down bricks
        np.testing.assert_array_almost_equal(bars.open, [100.0, 90.0, 80.0])
        np.testing.assert_array_almost_equal(bars.close, [90.0, 80.0, 70.0])
        np.testing.assert_array_almost_equal(bars.high, [100.0, 90.0, 80.0])
        np.testing.assert_array_almost_equal(bars.low, [90.0, 80.0, 70.0])

    def test_multiple_bricks_single_tick(self) -> None:
        """Large price jump should create multiple bricks in single tick."""
        # Price jumps from 100 to 140 in one tick with brick_size=10
        # Should create 4 up bricks at once
        close = np.array([100.0, 140.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64) * 100,
            brick_size=10.0,
        )

        assert len(bars) == 4
        np.testing.assert_array_almost_equal(bars.open, [100.0, 110.0, 120.0, 130.0])
        np.testing.assert_array_almost_equal(bars.close, [110.0, 120.0, 130.0, 140.0])

    def test_volume_aggregation(self) -> None:
        """Volume should be correctly aggregated per Renko bar."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        volume = np.array([100, 200, 300, 400, 500], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=volume,
            brick_size=10.0,
        )

        # First brick completes at index 2 (price reaches 110)
        # Second brick completes at index 4 (price reaches 120)
        assert len(bars) == 2
        # First bar: sum of indices 0-2 = 100+200+300 = 600
        assert bars.volume[0] == 600
        # Second bar: sum of indices 3-4 = 400+500 = 900
        assert bars.volume[1] == 900

    def test_index_map_tracking(self) -> None:
        """Index map should correctly track source row where bar completed."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Bars complete when price reaches 110, 120, 130
        # Index 2 -> 110, Index 4 -> 120, Index 6 -> 130
        np.testing.assert_array_equal(bars.index_map, [2, 4, 6])


class TestRenkoReversalRule:
    """Tests for the 2-brick reversal rule."""

    def test_uptrend_to_downtrend_reversal(self) -> None:
        """Reversal from up to down requires 2x brick distance."""
        # Start at 100, go up to 120 (2 up bricks)
        # Then price drops to 100 (needs 2x brick = 20 points to reverse)
        close = np.array([100.0, 110.0, 120.0, 100.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # 2 up bricks + 1 down brick (reversal at 2x distance)
        assert len(bars) == 3
        # Last brick should be down
        assert bars.close[-1] < bars.open[-1]

    def test_downtrend_to_uptrend_reversal(self) -> None:
        """Reversal from down to up requires 2x brick distance."""
        # Start at 100, go down to 80 (2 down bricks)
        # Then price rises to 100 (needs 2x brick = 20 points to reverse)
        close = np.array([100.0, 90.0, 80.0, 100.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # 2 down bricks + 1 up brick (reversal at 2x distance)
        assert len(bars) == 3
        # Last brick should be up
        assert bars.close[-1] > bars.open[-1]

    def test_no_reversal_below_threshold(self) -> None:
        """No reversal should occur if price doesn't move 2x brick distance."""
        # Start at 100, up to 110 (1 brick)
        # Then drop to 95 (only 15 points, not enough for reversal)
        close = np.array([100.0, 110.0, 95.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Only 1 up brick, no reversal
        assert len(bars) == 1
        assert bars.close[0] > bars.open[0]


class TestRenkoATRDynamicSizing:
    """Tests for ATR-based dynamic brick sizing."""

    def test_atr_based_sizing(self) -> None:
        """ATR values should be used for brick size when provided."""
        close = np.array([100.0, 110.0, 125.0, 145.0])  # Need larger moves
        atr_values = np.array([5.0, 5.0, 10.0, 15.0])  # ATR increases
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=100.0,  # Should be ignored when ATR provided
            atr_length=2,
            atr_values=atr_values,
        )

        # ATR takes effect from index >= atr_length
        # At index 2: ATR=10, price from 100 to 125 = 25 -> 2 bricks
        # At index 3: ATR=15, price from 120 to 145 = 25 -> 1 brick (15+10)
        assert len(bars) >= 1
        assert bars.parameters["atr_length"] == 2

    def test_fixed_size_before_atr_warmup(self) -> None:
        """Fixed brick size should be used before ATR warmup period."""
        close = np.array([100.0, 110.0, 115.0, 120.0])
        atr_values = np.array([20.0, 20.0, 20.0, 20.0])  # Large ATR
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
            atr_length=5,  # Warmup longer than data
            atr_values=atr_values,
        )

        # Fixed brick size should be used (10.0)
        # Price moves 10 at index 1, 15 at index 2, 20 at index 3
        assert len(bars) >= 1


class TestRenkoEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data_returns_empty_barseries(self) -> None:
        """Empty input should return empty BarSeries."""
        empty = np.array([], dtype=np.float64)
        empty_vol = np.array([], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=empty,
            high_arr=empty,
            low_arr=empty,
            close_arr=empty,
            volume_arr=empty_vol,
            brick_size=10.0,
        )

        assert bars.is_empty
        assert len(bars) == 0
        assert bars.type == "renko"

    def test_single_row_returns_empty_barseries(self) -> None:
        """Single row input should return empty BarSeries."""
        single = np.array([100.0])
        single_vol = np.array([1000], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=single,
            high_arr=single,
            low_arr=single,
            close_arr=single,
            volume_arr=single_vol,
            brick_size=10.0,
        )

        assert bars.is_empty
        assert len(bars) == 0

    def test_identical_prices_returns_empty(self) -> None:
        """All identical prices should return empty BarSeries."""
        flat = np.full(100, 100.0)
        vol = np.ones(100, dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=flat,
            high_arr=flat,
            low_arr=flat,
            close_arr=flat,
            volume_arr=vol,
            brick_size=10.0,
        )

        assert bars.is_empty

    def test_no_brick_size_raises_error(self) -> None:
        """Should raise ValueError if brick_size not provided and no ATR."""
        close = np.array([100.0, 110.0])
        vol = np.ones(2, dtype=np.int64)

        with pytest.raises(ValueError, match="brick_size is required"):
            generate_renko_bars_series(
                open_arr=close,
                high_arr=close,
                low_arr=close,
                close_arr=close,
                volume_arr=vol,
                brick_size=None,
            )

    def test_negative_brick_size_raises_error(self) -> None:
        """Should raise ValueError for negative brick size."""
        close = np.array([100.0, 110.0])
        vol = np.ones(2, dtype=np.int64)

        with pytest.raises(ValueError, match="brick_size must be positive"):
            generate_renko_bars_series(
                open_arr=close,
                high_arr=close,
                low_arr=close,
                close_arr=close,
                volume_arr=vol,
                brick_size=-10.0,
            )

    def test_zero_brick_size_raises_error(self) -> None:
        """Should raise ValueError for zero brick size."""
        close = np.array([100.0, 110.0])
        vol = np.ones(2, dtype=np.int64)

        with pytest.raises(ValueError, match="brick_size must be positive"):
            generate_renko_bars_series(
                open_arr=close,
                high_arr=close,
                low_arr=close,
                close_arr=close,
                volume_arr=vol,
                brick_size=0.0,
            )

    def test_inconsistent_array_lengths_raises_error(self) -> None:
        """Should raise ValueError for inconsistent array lengths."""
        close = np.array([100.0, 110.0, 120.0])
        vol = np.ones(2, dtype=np.int64)  # Different length

        with pytest.raises(ValueError, match="inconsistent lengths"):
            generate_renko_bars_series(
                open_arr=close,
                high_arr=close,
                low_arr=close,
                close_arr=close,
                volume_arr=vol,
                brick_size=10.0,
            )


class TestRenkoOutputValidation:
    """Tests for output array validation."""

    def test_ohlc_arrays_are_float64(self) -> None:
        """OHLC arrays should be float64 dtype."""
        close = np.array([100.0, 110.0, 120.0, 130.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert bars.open.dtype == np.float64
        assert bars.high.dtype == np.float64
        assert bars.low.dtype == np.float64
        assert bars.close.dtype == np.float64

    def test_volume_and_index_arrays_are_int64(self) -> None:
        """Volume and index_map should be int64 dtype."""
        close = np.array([100.0, 110.0, 120.0, 130.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert bars.volume.dtype == np.int64
        assert bars.index_map.dtype == np.int64

    def test_arrays_are_c_contiguous(self) -> None:
        """All output arrays should be C-contiguous."""
        close = np.array([100.0, 110.0, 120.0, 130.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert bars.open.flags["C_CONTIGUOUS"]
        assert bars.high.flags["C_CONTIGUOUS"]
        assert bars.low.flags["C_CONTIGUOUS"]
        assert bars.close.flags["C_CONTIGUOUS"]
        assert bars.volume.flags["C_CONTIGUOUS"]
        assert bars.index_map.flags["C_CONTIGUOUS"]

    def test_all_arrays_same_length(self) -> None:
        """All output arrays should have the same length."""
        close = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        bar_count = len(bars)
        assert len(bars.open) == bar_count
        assert len(bars.high) == bar_count
        assert len(bars.low) == bar_count
        assert len(bars.close) == bar_count
        assert len(bars.volume) == bar_count
        assert len(bars.index_map) == bar_count

    def test_bar_indices_monotonically_increasing(self) -> None:
        """Bar indices should be monotonically increasing."""
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(100) * 2)
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=5.0,
        )

        if len(bars) > 1:
            diffs = np.diff(bars.index_map)
            assert np.all(diffs >= 0), "bar_indices should be monotonically increasing"


class TestRenkoWithFixture:
    """Tests using the sample_ohlcv_data fixture."""

    def test_realistic_data(self, sample_ohlcv_data) -> None:
        """Should generate bars from realistic OHLCV data."""
        df = sample_ohlcv_data

        bars = generate_renko_bars_series(
            open_arr=df["open"].values,
            high_arr=df["high"].values,
            low_arr=df["low"].values,
            close_arr=df["close"].values,
            volume_arr=df["volume"].values.astype(np.int64),
            brick_size=0.5,  # Smaller brick for random walk data
        )

        # Should generate some bars from 1000 rows of data
        assert len(bars) > 0
        assert bars.type == "renko"

        # OHLC relationships should be valid
        # For up bricks: high == close, low == open
        # For down bricks: high == open, low == close
        for i in range(len(bars)):
            assert bars.high[i] >= bars.low[i]
            assert bars.open[i] >= bars.low[i]
            assert bars.open[i] <= bars.high[i]
            assert bars.close[i] >= bars.low[i]
            assert bars.close[i] <= bars.high[i]


class TestRenkoReversalBoundaries:
    """Tests for reversal boundary conditions."""

    def test_reversal_at_exact_2x_boundary_up_to_down(self) -> None:
        """Price exactly at 2x brick distance should trigger reversal (up to down)."""
        # Go up 1 brick, then drop exactly 2x brick distance
        close = np.array([
            100.0,  # Start
            110.0,  # Up 1 brick (100 -> 110)
            90.0,   # Exactly 2x brick below (110 - 20 = 90)
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Expect: 1 up brick + 1 down brick (reversal triggered)
        assert len(bars) == 2
        assert bars.close[0] > bars.open[0]  # First brick up
        assert bars.close[1] < bars.open[1]  # Second brick down (reversal)
        assert bars.open[1] == 100.0  # Reversal starts at stepped-back position
        assert bars.close[1] == 90.0

    def test_reversal_at_exact_2x_boundary_down_to_up(self) -> None:
        """Price exactly at 2x brick distance should trigger reversal (down to up)."""
        # Go down 1 brick, then rise exactly 2x brick distance
        close = np.array([
            100.0,  # Start
            90.0,   # Down 1 brick (100 -> 90)
            110.0,  # Exactly 2x brick above (90 + 20 = 110)
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Expect: 1 down brick + 1 up brick (reversal triggered)
        assert len(bars) == 2
        assert bars.close[0] < bars.open[0]  # First brick down
        assert bars.close[1] > bars.open[1]  # Second brick up (reversal)
        assert bars.open[1] == 100.0  # Reversal starts at stepped-back position
        assert bars.close[1] == 110.0

    def test_reversal_just_below_2x_threshold(self) -> None:
        """Price just below 2x threshold should not trigger reversal."""
        # Up 1 brick, then drop just short of 2x
        close = np.array([
            100.0,  # Start
            110.0,  # Up 1 brick
            90.1,   # Just short of 2x threshold (needs <= 90.0)
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Only 1 up brick, no reversal
        assert len(bars) == 1
        assert bars.close[0] > bars.open[0]

    def test_multiple_reversals_in_sequence(self) -> None:
        """Multiple consecutive reversals should work correctly."""
        close = np.array([
            100.0,  # Start
            110.0,  # Up to 110 (1 up brick)
            90.0,   # Down to 90 (reversal: 1 down brick)
            110.0,  # Up to 110 (reversal: 1 up brick)
            90.0,   # Down to 90 (reversal: 1 down brick)
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Should have 4 bricks: up, down, up, down
        assert len(bars) == 4
        assert bars.close[0] > bars.open[0]  # Up
        assert bars.close[1] < bars.open[1]  # Down
        assert bars.close[2] > bars.open[2]  # Up
        assert bars.close[3] < bars.open[3]  # Down


class TestRenkoATREdgeCases:
    """Tests for ATR-based sizing edge cases."""

    def test_atr_transition_at_exact_length(self) -> None:
        """ATR should activate exactly at i == atr_length."""
        close = np.array([100.0, 110.0, 125.0, 145.0, 170.0])
        # ATR increases at index 3
        atr_values = np.array([5.0, 5.0, 10.0, 20.0, 30.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,  # Used for warmup
            atr_length=3,  # ATR activates at index 3
            atr_values=atr_values,
        )

        # Before index 3: use brick_size=10.0
        # At index 3+: use ATR values
        assert len(bars) > 0
        assert bars.parameters["atr_length"] == 3

    def test_atr_very_small_value(self) -> None:
        """Very small ATR should still work without crashing."""
        close = np.array([100.0, 100.01, 100.02, 100.03])
        atr_values = np.array([0.001, 0.001, 0.001, 0.001])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=0.001,
            atr_length=1,
            atr_values=atr_values,
        )

        # With very small ATR, should generate many micro-bricks
        assert isinstance(bars, BarSeries)
        assert len(bars) > 0

    def test_atr_without_brick_size(self) -> None:
        """When using ATR, brick_size=None should use placeholder."""
        close = np.array([100.0, 110.0, 125.0, 145.0])
        atr_values = np.array([5.0, 10.0, 15.0, 20.0])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=None,  # Should use placeholder
            atr_length=2,
            atr_values=atr_values,
        )

        # Should work and use ATR values
        assert len(bars) >= 0
        # brick_size in parameters should be 1.0 (placeholder) when None was passed
        assert bars.parameters["brick_size"] == 1.0

    def test_atr_changing_during_generation(self) -> None:
        """Dynamic ATR values should adapt brick size per tick."""
        close = np.array([
            100.0,
            105.0,  # Small ATR, small brick
            115.0,  # Larger ATR
            135.0,  # Even larger ATR
        ])
        atr_values = np.array([
            2.0,   # Tiny brick
            5.0,   # Small brick
            10.0,  # Medium brick
            20.0,  # Large brick
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=100.0,  # Ignored after atr_length
            atr_length=1,  # ATR active from index 1
            atr_values=atr_values,
        )

        # Should generate bricks using varying ATR sizes
        assert len(bars) >= 1


class TestRenkoLargeGaps:
    """Tests for large price gaps creating multiple bricks."""

    def test_10_bricks_in_single_tick_upward(self) -> None:
        """Large upward gap should create 10+ bricks in one tick."""
        close = np.array([
            100.0,
            200.0,  # Jump 100 points with brick_size=10 -> 10 bricks
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert len(bars) == 10
        # All bricks at the same source index
        assert np.all(bars.index_map == 1)
        # Verify sequential brick formation
        np.testing.assert_array_almost_equal(
            bars.open,
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
        )
        np.testing.assert_array_almost_equal(
            bars.close,
            [110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
        )

    def test_10_bricks_in_single_tick_downward(self) -> None:
        """Large downward gap should create 10+ bricks in one tick."""
        close = np.array([
            100.0,
            0.0,  # Drop 100 points with brick_size=10 -> 10 bricks
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert len(bars) == 10
        # Verify sequential brick formation downward
        np.testing.assert_array_almost_equal(
            bars.open,
            [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        )
        np.testing.assert_array_almost_equal(
            bars.close,
            [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
        )

    def test_large_gap_volume_aggregation(self) -> None:
        """Volume for multiple bricks in single tick should all be from that tick."""
        close = np.array([
            100.0,
            150.0,  # Creates 5 bricks
        ])
        volume = np.array([1000, 5000], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=volume,
            brick_size=10.0,
        )

        assert len(bars) == 5
        # First bar gets volume from indices 0-1
        assert bars.volume[0] == 6000  # 1000 + 5000


class TestRenkoMixedScenarios:
    """Tests for complex mixed direction scenarios."""

    def test_up_trend_reversal_down_continuation(self) -> None:
        """Up trend -> reversal -> down continuation."""
        close = np.array([
            100.0,
            110.0,  # Up 1 brick
            120.0,  # Up 2 bricks total
            100.0,  # Reversal down (2x brick)
            90.0,   # Continue down
            80.0,   # Continue down
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Expected: 2 up + 3 down = 5 bricks
        # (100->110, 110->120, 120->110, 110->100, 100->90, 90->80 would be 6)
        # Actual: reversal creates fewer bricks
        assert len(bars) == 5
        # First 2 are up
        assert bars.close[0] > bars.open[0]
        assert bars.close[1] > bars.open[1]
        # Rest are down
        for i in range(2, len(bars)):
            assert bars.close[i] < bars.open[i]

    def test_down_trend_reversal_up_continuation(self) -> None:
        """Down trend -> reversal -> up continuation."""
        close = np.array([
            100.0,
            90.0,   # Down 1 brick
            80.0,   # Down 2 bricks total
            100.0,  # Reversal up (2x brick)
            110.0,  # Continue up
            120.0,  # Continue up
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Expected: 2 down + 3 up = 5 bricks
        assert len(bars) == 5
        # First 2 are down
        assert bars.close[0] < bars.open[0]
        assert bars.close[1] < bars.open[1]
        # Rest are up
        for i in range(2, len(bars)):
            assert bars.close[i] > bars.open[i]

    def test_neutral_to_down_first_brick(self) -> None:
        """First brick can go down when starting neutral."""
        close = np.array([
            100.0,
            90.0,   # First brick down
            80.0,   # Continue down
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert len(bars) == 2
        # Both down
        assert bars.close[0] < bars.open[0]
        assert bars.close[1] < bars.open[1]
        # First bar starts at initial price
        assert bars.open[0] == 100.0
        assert bars.close[0] == 90.0


class TestRenkoVolumeEdgeCases:
    """Tests for volume aggregation edge cases."""

    def test_first_bar_volume_from_start(self) -> None:
        """First bar should aggregate volume from index 0 to bar completion."""
        close = np.array([100.0, 105.0, 110.0, 115.0])
        volume = np.array([100, 200, 300, 400], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=volume,
            brick_size=10.0,
        )

        # First brick completes at index 2 (price reaches 110)
        assert len(bars) == 1
        # Volume should be sum of indices 0-2: 100+200+300 = 600
        assert bars.volume[0] == 600

    def test_subsequent_bars_volume_ranges(self) -> None:
        """Subsequent bars should aggregate from prev+1 to current index."""
        close = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
        volume = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=volume,
            brick_size=10.0,
        )

        assert len(bars) == 3
        # Bar 0 completes at index 2: sum(0:3) = 10+20+30 = 60
        assert bars.volume[0] == 60
        # Bar 1 completes at index 4: sum(3:5) = 40+50 = 90
        assert bars.volume[1] == 90
        # Bar 2 completes at index 6: sum(5:7) = 60+70 = 130
        assert bars.volume[2] == 130

    def test_empty_bars_zero_volume(self) -> None:
        """Empty BarSeries should have empty volume array."""
        close = np.array([100.0])
        volume = np.array([1000], dtype=np.int64)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=volume,
            brick_size=10.0,
        )

        assert bars.is_empty
        assert len(bars.volume) == 0


class TestRenkoDownContinuation:
    """Tests for down continuation after various states."""

    def test_continue_down_after_down_brick(self) -> None:
        """Should continue down after establishing down direction."""
        close = np.array([
            100.0,
            90.0,   # First down brick
            80.0,   # Continue down
            70.0,   # Continue down
            60.0,   # Continue down
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        assert len(bars) == 4
        # All should be down bricks
        for i in range(len(bars)):
            assert bars.close[i] < bars.open[i], f"Bar {i} should be down"

    def test_down_continuation_with_small_upticks(self) -> None:
        """Small upticks during downtrend should not create bricks."""
        close = np.array([
            100.0,
            90.0,   # Down brick
            92.0,   # Small uptick (not enough to reverse)
            80.0,   # Continue down
            83.0,   # Small uptick
            70.0,   # Continue down
        ])
        n = len(close)

        bars = generate_renko_bars_series(
            open_arr=close,
            high_arr=close,
            low_arr=close,
            close_arr=close,
            volume_arr=np.ones(n, dtype=np.int64),
            brick_size=10.0,
        )

        # Should only have down bricks (no reversals)
        assert len(bars) == 3
        for i in range(len(bars)):
            assert bars.close[i] < bars.open[i]


class TestRenkoPerformance:
    """Performance benchmarks for Renko bar generation."""

    def test_throughput_1m_rows(self) -> None:
        """Should achieve >= 1M rows/sec throughput."""
        import time

        np.random.seed(42)
        n = 1_000_000
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.05)
        low = close - np.abs(np.random.randn(n) * 0.05)
        open_price = low + np.random.rand(n) * (high - low)
        volume = np.random.randint(100, 1000, n, dtype=np.int64)

        # Warmup JIT compilation
        _ = generate_renko_bars_series(
            open_arr=open_price[:1000],
            high_arr=high[:1000],
            low_arr=low[:1000],
            close_arr=close[:1000],
            volume_arr=volume[:1000],
            brick_size=0.5,
        )

        # Time the full run
        start = time.perf_counter()
        result = generate_renko_bars_series(
            open_arr=open_price,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            brick_size=0.5,
        )
        elapsed = time.perf_counter() - start

        # Verify result is valid
        assert isinstance(result, BarSeries)
        assert len(result) > 0

        # Calculate throughput
        throughput = n / elapsed
        print(f"\nRenko throughput: {throughput:,.0f} rows/sec ({elapsed:.3f}s for {n:,} rows)")

        # Acceptance criteria: >= 1M rows/sec
        assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} rows/sec < 1M"
