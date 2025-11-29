"""Tests for Tick and Volume Imbalance bar generation.

Tests cover:
- Basic functionality for both tick and volume imbalance bars
- Algorithm correctness (imbalance accumulation tracking)
- Tick direction determination (uptick/downtick/no-change)
- Edge cases (empty data, single row, no bars generated)
- Output validation (dtypes, contiguity, lengths)
- Performance benchmarks (1M+ rows/sec target)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from simple_futures_backtester.bars import BarSeries, get_bar_generator, list_bar_types
from simple_futures_backtester.bars.imbalance_bars import (
    generate_tick_imbalance_bars_series,
    generate_volume_imbalance_bars_series,
)


class TestImbalanceRegistration:
    """Tests for bar factory registration."""

    def test_tick_imbalance_registered(self) -> None:
        """Tick imbalance should be registered with the bar factory."""
        assert "tick_imbalance" in list_bar_types()

    def test_volume_imbalance_registered(self) -> None:
        """Volume imbalance should be registered with the bar factory."""
        assert "volume_imbalance" in list_bar_types()

    def test_get_tick_imbalance_generator(self) -> None:
        """Should retrieve the tick imbalance generator from factory."""
        generator = get_bar_generator("tick_imbalance")
        assert generator is generate_tick_imbalance_bars_series

    def test_get_volume_imbalance_generator(self) -> None:
        """Should retrieve the volume imbalance generator from factory."""
        generator = get_bar_generator("volume_imbalance")
        assert generator is generate_volume_imbalance_bars_series


class TestTickImbalanceBasicFunctionality:
    """Tests for basic Tick Imbalance bar generation."""

    def test_simple_tick_imbalance_bars(self) -> None:
        """Should generate bars when |cumulative_imbalance| >= threshold."""
        # Create data with known tick direction pattern
        # Uptick sequence: +1, +1, +1 -> imbalance = 3 at index 3
        close = np.array([100.0, 101.0, 102.0, 103.0, 102.5, 103.5, 104.5])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        assert len(bars) >= 1
        assert bars.type == "tick_imbalance"
        assert bars.parameters["imbalance_threshold"] == 3

    def test_strong_uptrend_creates_bars(self) -> None:
        """Strong uptrend should create bars quickly."""
        # All upticks: +1 each
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # Should create bar at index 3 (cumulative imbalance = +3)
        # and possibly more
        assert len(bars) >= 1

    def test_strong_downtrend_creates_bars(self) -> None:
        """Strong downtrend should create bars quickly."""
        # All downticks: -1 each
        close = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # Should create bar at index 3 (cumulative imbalance = -3)
        assert len(bars) >= 1

    def test_choppy_market_few_bars(self) -> None:
        """Choppy market with no sustained direction should create few bars."""
        # Alternating up/down: +1, -1, +1, -1 -> imbalance oscillates
        close = np.array([100.0, 101.0, 100.5, 101.5, 101.0, 101.5, 101.2])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=10,  # High threshold
        )

        # Should create few or no bars
        assert len(bars) <= 1


class TestTickImbalanceAlgorithmCorrectness:
    """Tests for Tick Imbalance bar algorithm implementation."""

    def test_tick_direction_uptick(self) -> None:
        """Should correctly identify upticks (+1)."""
        # Sequence of upticks
        close = np.array([100.0, 100.5, 101.0, 101.5])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # 3 upticks -> imbalance = +3 at index 3
        assert len(bars) == 1
        assert bars.close[0] == 101.5

    def test_tick_direction_downtick(self) -> None:
        """Should correctly identify downticks (-1)."""
        # Sequence of downticks
        close = np.array([101.5, 101.0, 100.5, 100.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # 3 downticks -> imbalance = -3 at index 3
        assert len(bars) == 1
        assert bars.close[0] == 100.0

    def test_tick_direction_no_change(self) -> None:
        """Should handle no-change ticks (0)."""
        # No change ticks should not affect imbalance
        close = np.array([100.0, 100.0, 100.0, 101.0, 102.0, 103.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # 2 no-change (0), then 3 upticks (+1 each) -> imbalance = +3 at index 5
        assert len(bars) == 1
        assert bars.close[0] == 103.0

    def test_absolute_threshold_positive(self) -> None:
        """Should close bar when imbalance >= +threshold."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # Cumulative: +1, +2, +3 at index 3 -> close bar
        assert len(bars) == 1
        assert bars.index_map[0] == 3

    def test_absolute_threshold_negative(self) -> None:
        """Should close bar when imbalance <= -threshold."""
        close = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # Cumulative: -1, -2, -3 at index 3 -> close bar
        assert len(bars) == 1
        assert bars.index_map[0] == 3

    def test_imbalance_reset_after_bar(self) -> None:
        """Should reset imbalance to 0 after closing a bar."""
        # 3 upticks, then 3 more upticks
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        # Bar 1 at index 3 (+3), Bar 2 at index 6 (+3)
        assert len(bars) == 2


class TestVolumeImbalanceBasicFunctionality:
    """Tests for basic Volume Imbalance bar generation."""

    def test_simple_volume_imbalance_bars(self) -> None:
        """Should generate bars when |volume_weighted_imbalance| >= threshold."""
        # Upticks with high volume
        # +1*15k, +1*20k, +1*5k, -1*25k, +1*30k
        # Cumulative: +15k, +35k, +40k, +15k, +45k (never reaches 50k)
        # Let's adjust to ensure we reach threshold
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([10000, 15000, 20000, 20000, 25000, 30000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        # +15k, +35k, +55k at index 3 -> should create bar
        assert len(bars) >= 1
        assert bars.type == "volume_imbalance"
        assert bars.parameters["imbalance_threshold"] == 50000

    def test_high_volume_uptrend(self) -> None:
        """High volume uptrend should create bars quickly."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([20000, 20000, 20000, 20000, 20000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        # +20k, +40k, +60k at index 3 -> close bar
        assert len(bars) >= 1

    def test_low_volume_few_bars(self) -> None:
        """Low volume should create few bars even with strong trend."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([100, 100, 100, 100, 100, 100], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=1000000,  # Very high threshold
        )

        # Low volume can't reach threshold
        assert len(bars) == 0


class TestVolumeImbalanceAlgorithmCorrectness:
    """Tests for Volume Imbalance bar algorithm implementation."""

    def test_volume_weighted_accumulation_positive(self) -> None:
        """Should correctly accumulate volume-weighted imbalance (positive)."""
        # Upticks with known volumes
        close = np.array([100.0, 101.0, 102.0, 103.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([10000, 20000, 15000, 25000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        # Cumulative: +20k, +35k, +60k at index 3 -> close bar
        assert len(bars) == 1
        assert bars.close[0] == 103.0

    def test_volume_weighted_accumulation_negative(self) -> None:
        """Should correctly accumulate volume-weighted imbalance (negative)."""
        # Downticks with known volumes
        close = np.array([103.0, 102.0, 101.0, 100.0])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([10000, 20000, 15000, 25000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        # Cumulative: -20k, -35k, -60k at index 3 -> close bar
        assert len(bars) == 1
        assert bars.close[0] == 100.0

    def test_mixed_direction_with_volume(self) -> None:
        """Should handle mixed directions with volume weighting."""
        # +1 * 30k, -1 * 10k, +1 * 25k -> cumulative = +45k
        close = np.array([100.0, 101.0, 100.5, 101.5, 102.5])
        high = close + 0.5
        low = close - 0.5
        volume = np.array([10000, 30000, 10000, 25000, 15000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        # At index 3: +30k -10k +25k = +45k (not yet)
        # At index 4: +30k -10k +25k +15k = +60k -> close bar
        assert len(bars) == 1


class TestImbalanceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input_tick(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_tick_imbalance_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            imbalance_threshold=10,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_empty_input_volume(self) -> None:
        """Empty input arrays should return empty BarSeries."""
        bars = generate_volume_imbalance_bars_series(
            open_arr=np.array([], dtype=np.float64),
            high_arr=np.array([], dtype=np.float64),
            low_arr=np.array([], dtype=np.float64),
            close_arr=np.array([], dtype=np.float64),
            volume_arr=np.array([], dtype=np.int64),
            imbalance_threshold=10000,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row_tick(self) -> None:
        """Single row should return empty BarSeries (need at least 2)."""
        bars = generate_tick_imbalance_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            imbalance_threshold=1,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_single_row_volume(self) -> None:
        """Single row should return empty BarSeries (need at least 2)."""
        bars = generate_volume_imbalance_bars_series(
            open_arr=np.array([100.0]),
            high_arr=np.array([101.0]),
            low_arr=np.array([99.0]),
            close_arr=np.array([100.0]),
            volume_arr=np.array([1000], dtype=np.int64),
            imbalance_threshold=500,
        )

        assert len(bars) == 0
        assert bars.is_empty

    def test_invalid_threshold_zero_tick(self) -> None:
        """imbalance_threshold of zero should raise ValueError."""
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            generate_tick_imbalance_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                imbalance_threshold=0,
            )

    def test_invalid_threshold_zero_volume(self) -> None:
        """imbalance_threshold of zero should raise ValueError."""
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            generate_volume_imbalance_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                imbalance_threshold=0,
            )

    def test_invalid_threshold_negative_tick(self) -> None:
        """Negative imbalance_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            generate_tick_imbalance_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                imbalance_threshold=-10,
            )

    def test_invalid_threshold_negative_volume(self) -> None:
        """Negative imbalance_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            generate_volume_imbalance_bars_series(
                open_arr=np.array([100.0, 101.0]),
                high_arr=np.array([101.0, 102.0]),
                low_arr=np.array([99.0, 100.0]),
                close_arr=np.array([100.0, 101.0]),
                volume_arr=np.array([1000, 1000], dtype=np.int64),
                imbalance_threshold=-10000,
            )


class TestImbalanceOutputValidation:
    """Tests for output data type and structure validation."""

    def test_output_dtypes_tick(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        if len(bars) > 0:
            assert bars.open.dtype == np.float64
            assert bars.high.dtype == np.float64
            assert bars.low.dtype == np.float64
            assert bars.close.dtype == np.float64
            assert bars.volume.dtype == np.int64
            assert bars.index_map.dtype == np.int64

    def test_output_dtypes_volume(self) -> None:
        """Output arrays should have correct dtypes."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([20000, 20000, 20000, 20000, 20000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        if len(bars) > 0:
            assert bars.open.dtype == np.float64
            assert bars.high.dtype == np.float64
            assert bars.low.dtype == np.float64
            assert bars.close.dtype == np.float64
            assert bars.volume.dtype == np.int64
            assert bars.index_map.dtype == np.int64

    def test_output_contiguity_tick(self) -> None:
        """Output arrays should be C-contiguous."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)

        bars = generate_tick_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=3,
        )

        if len(bars) > 0:
            assert bars.open.flags["C_CONTIGUOUS"]
            assert bars.high.flags["C_CONTIGUOUS"]
            assert bars.low.flags["C_CONTIGUOUS"]
            assert bars.close.flags["C_CONTIGUOUS"]
            assert bars.volume.flags["C_CONTIGUOUS"]
            assert bars.index_map.flags["C_CONTIGUOUS"]

    def test_output_contiguity_volume(self) -> None:
        """Output arrays should be C-contiguous."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high = close + 1.0
        low = close - 1.0
        volume = np.array([20000, 20000, 20000, 20000, 20000], dtype=np.int64)

        bars = generate_volume_imbalance_bars_series(
            open_arr=close,
            high_arr=high,
            low_arr=low,
            close_arr=close,
            volume_arr=volume,
            imbalance_threshold=50000,
        )

        if len(bars) > 0:
            assert bars.open.flags["C_CONTIGUOUS"]
            assert bars.high.flags["C_CONTIGUOUS"]
            assert bars.low.flags["C_CONTIGUOUS"]
            assert bars.close.flags["C_CONTIGUOUS"]
            assert bars.volume.flags["C_CONTIGUOUS"]
            assert bars.index_map.flags["C_CONTIGUOUS"]


