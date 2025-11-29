"""Comprehensive tests for data validation.

Tests cover:
- Custom exception DataValidationError attributes
- NaN/null detection in all OHLCV columns
- Timestamp monotonicity validation
- OHLC bounds validation
- Volume non-negativity validation
- Full validation pipeline integration
- Fail-fast behavior verification
- Edge cases (empty data, single row, all valid)

Coverage Target: 85%+ for simple_futures_backtester.data.validation
"""

import numpy as np
import pandas as pd
import pytest

from simple_futures_backtester.data.validation import (
    DataValidationError,
    validate_data,
    _check_nan_values,
    _check_timestamp_monotonicity,
    _check_ohlc_bounds,
    _check_volume_non_negative,
)


def create_valid_ohlcv_df(n_rows: int = 3) -> pd.DataFrame:
    """Create a valid OHLCV DataFrame for testing.

    Args:
        n_rows: Number of rows to generate.

    Returns:
        DataFrame with valid datetime, OHLCV data.
    """
    return pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n_rows, freq='1min'),
        'open': [100.0 + i for i in range(n_rows)],
        'high': [100.5 + i for i in range(n_rows)],
        'low': [99.5 + i for i in range(n_rows)],
        'close': [100.25 + i for i in range(n_rows)],
        'volume': [1000] * n_rows
    })


class TestDataValidationError:
    """Tests for DataValidationError custom exception."""

    def test_exception_attributes(self):
        """DataValidationError should store diagnostic attributes."""
        exc = DataValidationError(
            row_index=42,
            column='close',
            value=np.nan,
            message='Test error'
        )

        assert exc.row_index == 42
        assert exc.column == 'close'
        assert exc.message == 'Test error'

    def test_exception_message_format(self):
        """Exception message should include row index and description."""
        exc = DataValidationError(
            row_index=10,
            column='high',
            value=99.5,
            message='Invalid OHLC relationship'
        )

        error_str = str(exc)
        assert 'row 10' in error_str.lower()
        assert 'Invalid OHLC relationship' in error_str


class TestCheckNanValues:
    """Tests for _check_nan_values function."""

    @pytest.mark.parametrize("column", ["datetime", "open", "high", "low", "close"])
    def test_nan_in_column_raises_error(self, column):
        """NaN in any OHLC column should raise DataValidationError."""
        df = create_valid_ohlcv_df()
        df.loc[1, column] = np.nan if column != 'datetime' else pd.NaT

        with pytest.raises(DataValidationError) as exc_info:
            _check_nan_values(df)

        assert exc_info.value.column == column
        assert exc_info.value.row_index == 1
        assert f"NaN/null value detected in '{column}' column" in str(exc_info.value)

    def test_first_nan_row_reported(self):
        """Should report the first NaN found in column scan order."""
        df = create_valid_ohlcv_df(5)
        # NaN checking is per-column, so it finds first NaN in 'close' column
        df.loc[1, 'close'] = np.nan
        df.loc[3, 'close'] = np.nan

        with pytest.raises(DataValidationError) as exc_info:
            _check_nan_values(df)

        # Should catch the first NaN in 'close' column at row 1
        assert exc_info.value.row_index == 1
        assert exc_info.value.column == 'close'

    def test_all_valid_passes(self):
        """All valid OHLC data should pass without raising."""
        df = create_valid_ohlcv_df()

        # Should not raise
        _check_nan_values(df)

    def test_nan_in_datetime_column_first(self):
        """NaN in datetime should be caught first (datetime checked before other columns)."""
        df = create_valid_ohlcv_df()
        df.loc[0, 'datetime'] = pd.NaT

        with pytest.raises(DataValidationError) as exc_info:
            _check_nan_values(df)

        assert exc_info.value.column == 'datetime'
        assert exc_info.value.row_index == 0


class TestCheckTimestampMonotonicity:
    """Tests for _check_timestamp_monotonicity function."""

    def test_duplicate_timestamps_fail(self):
        """Duplicate timestamps should raise DataValidationError."""
        df = create_valid_ohlcv_df()
        df.loc[2, 'datetime'] = df.loc[1, 'datetime']

        with pytest.raises(DataValidationError) as exc_info:
            _check_timestamp_monotonicity(df)

        assert 'not monotonically increasing' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 2

    def test_backwards_timestamps_fail(self):
        """Backwards timestamps should raise DataValidationError."""
        df = create_valid_ohlcv_df()
        df.loc[2, 'datetime'] = df.loc[0, 'datetime'] - pd.Timedelta(minutes=1)

        with pytest.raises(DataValidationError) as exc_info:
            _check_timestamp_monotonicity(df)

        assert 'not monotonically increasing' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 2

    def test_single_row_passes(self):
        """Single row data should pass (no comparison possible)."""
        df = create_valid_ohlcv_df(n_rows=1)

        # Should not raise
        _check_timestamp_monotonicity(df)

    def test_strictly_increasing_passes(self):
        """Strictly increasing timestamps should pass."""
        df = create_valid_ohlcv_df()

        # Should not raise
        _check_timestamp_monotonicity(df)


class TestCheckOhlcBounds:
    """Tests for _check_ohlc_bounds function."""

    def test_high_less_than_open_fails(self):
        """high < open should raise DataValidationError when open > close."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 105.0
        df.loc[1, 'high'] = 104.0  # high < open
        df.loc[1, 'close'] = 103.0
        df.loc[1, 'low'] = 102.0

        with pytest.raises(DataValidationError) as exc_info:
            _check_ohlc_bounds(df)

        assert 'high' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1

    def test_high_less_than_close_fails(self):
        """high < close should raise DataValidationError when close > open."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'close'] = 105.0
        df.loc[1, 'high'] = 104.0  # high < close
        df.loc[1, 'low'] = 99.0

        with pytest.raises(DataValidationError) as exc_info:
            _check_ohlc_bounds(df)

        assert 'high' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1

    def test_low_greater_than_open_fails(self):
        """low > open should raise DataValidationError when open < close."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'close'] = 105.0
        df.loc[1, 'high'] = 106.0
        df.loc[1, 'low'] = 101.0  # low > open

        with pytest.raises(DataValidationError) as exc_info:
            _check_ohlc_bounds(df)

        assert 'low' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1

    def test_low_greater_than_close_fails(self):
        """low > close should raise DataValidationError when close < open."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 105.0
        df.loc[1, 'close'] = 100.0
        df.loc[1, 'high'] = 106.0
        df.loc[1, 'low'] = 101.0  # low > close

        with pytest.raises(DataValidationError) as exc_info:
            _check_ohlc_bounds(df)

        assert 'low' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1

    def test_high_less_than_low_fails(self):
        """high < low should raise DataValidationError (impossible OHLC relationship).

        Note: The direct high < low check in validation.py (lines 146-154) is actually
        unreachable by design. If high < low, then either:
        - high < max(open, close) (caught by first check), or
        - low > min(open, close) (caught by second check)

        This test verifies that such invalid data IS caught (by one of the first two checks).
        """
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'close'] = 100.0
        df.loc[1, 'high'] = 99.0  # high=99 < low=100, and high < max(open, close)
        df.loc[1, 'low'] = 100.0

        with pytest.raises(DataValidationError) as exc_info:
            _check_ohlc_bounds(df)

        # Will be caught by first check: high < max(open, close)
        assert exc_info.value.row_index == 1
        assert 'high' in str(exc_info.value).lower()

    def test_valid_ohlc_relationships_pass(self):
        """Valid OHLC relationships should pass."""
        df = create_valid_ohlcv_df()

        # Should not raise
        _check_ohlc_bounds(df)

    def test_ohlc_all_equal_passes(self):
        """OHLC all equal (flat bar) should pass."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'high'] = 100.0
        df.loc[1, 'low'] = 100.0
        df.loc[1, 'close'] = 100.0

        # Should not raise
        _check_ohlc_bounds(df)


class TestCheckVolumeNonNegative:
    """Tests for _check_volume_non_negative function."""

    def test_negative_volume_fails(self):
        """Negative volume should raise DataValidationError."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'volume'] = -100

        with pytest.raises(DataValidationError) as exc_info:
            _check_volume_non_negative(df)

        assert 'volume' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1
        assert exc_info.value.value == -100

    def test_zero_volume_passes(self):
        """Zero volume should pass (non-negative includes zero)."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'volume'] = 0

        # Should not raise
        _check_volume_non_negative(df)

    def test_positive_volume_passes(self):
        """Positive volume should pass."""
        df = create_valid_ohlcv_df()

        # Should not raise
        _check_volume_non_negative(df)


class TestValidateDataIntegration:
    """Tests for validate_data full pipeline."""

    def test_all_valid_data_passes(self):
        """Completely valid data should pass all checks."""
        df = create_valid_ohlcv_df()

        result = validate_data(df)

        # Should return the same DataFrame
        assert result is df

    def test_validation_order_nan_first(self):
        """NaN check should run before other checks."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'close'] = np.nan  # NaN error
        df.loc[2, 'datetime'] = df.loc[1, 'datetime']  # Also has timestamp error

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should catch NaN first
        assert exc_info.value.column == 'close'
        assert 'NaN' in str(exc_info.value)

    def test_validation_order_timestamp_second(self):
        """Timestamp check should run after NaN check."""
        df = create_valid_ohlcv_df()
        df.loc[2, 'datetime'] = df.loc[1, 'datetime']  # Duplicate timestamp
        df.loc[1, 'high'] = 90.0  # Also has OHLC error (high < low)
        df.loc[1, 'low'] = 100.0

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should catch timestamp error before OHLC error
        assert 'not monotonically increasing' in str(exc_info.value).lower()

    def test_validation_order_ohlc_third(self):
        """OHLC check should run after timestamp check."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'high'] = 90.0  # OHLC error (high < low)
        df.loc[1, 'low'] = 100.0
        df.loc[2, 'volume'] = -100  # Also has volume error

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should catch OHLC error before volume error
        assert 'high' in str(exc_info.value).lower() or 'low' in str(exc_info.value).lower()
        assert exc_info.value.row_index == 1

    def test_validation_order_volume_last(self):
        """Volume check should run last."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'volume'] = -100  # Only volume error

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should catch volume error
        assert 'volume' in str(exc_info.value).lower()
        assert exc_info.value.value == -100

    def test_returns_same_dataframe(self):
        """validate_data should return the same DataFrame instance."""
        df = create_valid_ohlcv_df()

        result = validate_data(df)

        # Should return the exact same object (not a copy)
        assert result is df


class TestValidateDataEdgeCases:
    """Tests for edge cases and error messages."""

    def test_error_message_clarity_nan(self):
        """NaN error messages should be descriptive and actionable."""
        df = create_valid_ohlcv_df()
        df.loc[2, 'open'] = np.nan

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        error_msg = str(exc_info.value)
        assert 'row 2' in error_msg.lower()
        assert 'open' in error_msg
        assert 'NaN' in error_msg or 'null' in error_msg

    def test_error_message_clarity_timestamp(self):
        """Timestamp error messages should be descriptive."""
        df = create_valid_ohlcv_df()
        df.loc[2, 'datetime'] = df.loc[1, 'datetime']

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        error_msg = str(exc_info.value)
        assert 'row 2' in error_msg.lower()
        assert 'datetime' in error_msg.lower() or 'timestamp' in error_msg.lower()

    def test_error_message_clarity_ohlc(self):
        """OHLC error messages should be descriptive."""
        df = create_valid_ohlcv_df()
        df.loc[1, 'high'] = 90.0
        df.loc[1, 'low'] = 100.0

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        error_msg = str(exc_info.value)
        assert 'row 1' in error_msg.lower()

    def test_fail_fast_behavior(self):
        """Should stop on first error, not collect all errors."""
        df = create_valid_ohlcv_df(5)
        # Create multiple violations
        df.loc[0, 'close'] = np.nan  # First error
        df.loc[2, 'datetime'] = df.loc[1, 'datetime']  # Second error
        df.loc[3, 'high'] = 90.0  # Third error
        df.loc[3, 'low'] = 100.0
        df.loc[4, 'volume'] = -100  # Fourth error

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should only catch the first error (NaN at row 0)
        assert exc_info.value.row_index == 0
        assert exc_info.value.column == 'close'

    def test_empty_dataframe_handling(self):
        """Empty DataFrame should be handled gracefully."""
        df = pd.DataFrame({
            'datetime': pd.DatetimeIndex([]),
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })

        # Should not raise (no data to validate)
        result = validate_data(df)
        assert result is df

    def test_single_row_handling(self):
        """Single row data should pass all checks."""
        df = create_valid_ohlcv_df(n_rows=1)

        result = validate_data(df)

        # Should pass (single row is valid, no timestamp comparison needed)
        assert result is df

    def test_single_row_with_invalid_ohlc_fails(self):
        """Single row with invalid OHLC should fail."""
        df = create_valid_ohlcv_df(n_rows=1)
        df.loc[0, 'high'] = 90.0
        df.loc[0, 'low'] = 100.0

        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df)

        # Should catch OHLC error even for single row
        assert exc_info.value.row_index == 0
