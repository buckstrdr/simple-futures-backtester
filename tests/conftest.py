"""Pytest fixtures and configuration for the test suite.

Provides sample data fixtures, common test utilities, and benchmark
configuration for the Simple Futures Backtester test suite.
"""

import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_rows = 1000

    dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="1min")
    close = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.1)
    high = close + np.abs(np.random.randn(n_rows) * 0.05)
    low = close - np.abs(np.random.randn(n_rows) * 0.05)
    open_price = low + np.random.rand(n_rows) * (high - low)
    volume = np.random.randint(100, 1000, n_rows)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_tick_data():
    """Provide sample tick data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_ticks = 10000

    timestamps = pd.date_range(start="2023-01-01", periods=n_ticks, freq="100ms")
    prices = 100.0 + np.cumsum(np.random.randn(n_ticks) * 0.01)
    volumes = np.random.randint(1, 10, n_ticks)

    return pd.DataFrame(
        {
            "price": prices,
            "volume": volumes,
        },
        index=timestamps,
    )
