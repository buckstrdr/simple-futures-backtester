"""Pytest fixtures for output module tests.

Provides sample BacktestResult and SweepResult fixtures for testing
reports, charts, and exports modules.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from simple_futures_backtester.backtest.engine import BacktestResult
from simple_futures_backtester.backtest.sweep import SweepResult


@pytest.fixture
def sample_backtest_result() -> BacktestResult:
    """Create a sample BacktestResult for testing.

    Creates a BacktestResult with 365 days of equity data suitable for
    monthly heatmap testing, realistic metrics, and sample trades.

    Returns:
        BacktestResult with realistic test data spanning 365 days.
    """
    # Create equity curve with 365 days for monthly heatmap testing
    np.random.seed(42)
    n_bars = 365
    base_equity = 100000.0
    daily_returns = np.random.normal(0.0003, 0.02, n_bars)
    equity_curve = base_equity * np.cumprod(1 + daily_returns)

    # Calculate drawdown curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_curve = (equity_curve - running_max) / running_max

    # Create sample trades DataFrame with expected column structure
    n_trades = 10
    trades_data = {
        "Entry Time": pd.date_range("2024-01-01", periods=n_trades, freq="7D"),
        "Exit Time": pd.date_range("2024-01-02", periods=n_trades, freq="7D"),
        "Entry Price": np.random.uniform(100, 105, n_trades),
        "Exit Price": np.random.uniform(100, 105, n_trades),
        "PnL": np.random.uniform(-100, 200, n_trades),
        "Return": np.random.uniform(-0.01, 0.02, n_trades),
        "Duration": ["1 day"] * n_trades,
        "Direction": ["Long"] * 5 + ["Short"] * 5,
    }
    trades_df = pd.DataFrame(trades_data)

    return BacktestResult(
        total_return=0.1523,
        sharpe_ratio=1.2345,
        sortino_ratio=1.5678,
        max_drawdown=0.0823,
        win_rate=0.6234,
        profit_factor=1.8765,
        n_trades=10,
        avg_trade=125.67,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trades=trades_df,
        config_hash="abc123def456789012345678901234567890abcdef0123456789012345678",
        timestamp="2025-11-28T12:00:00+00:00",
    )


@pytest.fixture
def sample_sweep_result(sample_backtest_result: BacktestResult) -> SweepResult:
    """Create a sample SweepResult for testing.

    Creates a SweepResult with three parameter combinations for testing
    sweep report generation.

    Args:
        sample_backtest_result: Base BacktestResult fixture to use for each combo.

    Returns:
        SweepResult with three parameter combinations sorted by Sharpe.
    """
    all_results = []

    # Best result (highest Sharpe)
    params1 = {"stop_loss": 100, "take_profit": 200}
    result1 = BacktestResult(
        total_return=0.2500,
        sharpe_ratio=2.5000,
        sortino_ratio=2.8000,
        max_drawdown=0.0500,
        win_rate=0.7000,
        profit_factor=2.5000,
        n_trades=50,
        avg_trade=150.0,
        equity_curve=sample_backtest_result.equity_curve.copy(),
        drawdown_curve=sample_backtest_result.drawdown_curve.copy(),
        trades=sample_backtest_result.trades.copy(),
        config_hash=sample_backtest_result.config_hash,
        timestamp=sample_backtest_result.timestamp,
    )
    all_results.append((params1, result1))

    # Second best
    params2 = {"stop_loss": 150, "take_profit": 200}
    result2 = BacktestResult(
        total_return=0.2000,
        sharpe_ratio=2.0000,
        sortino_ratio=2.3000,
        max_drawdown=0.0600,
        win_rate=0.6500,
        profit_factor=2.2000,
        n_trades=45,
        avg_trade=140.0,
        equity_curve=sample_backtest_result.equity_curve.copy(),
        drawdown_curve=sample_backtest_result.drawdown_curve.copy(),
        trades=sample_backtest_result.trades.copy(),
        config_hash=sample_backtest_result.config_hash,
        timestamp=sample_backtest_result.timestamp,
    )
    all_results.append((params2, result2))

    # Third best
    params3 = {"stop_loss": 100, "take_profit": 150}
    result3 = BacktestResult(
        total_return=0.1500,
        sharpe_ratio=1.5000,
        sortino_ratio=1.8000,
        max_drawdown=0.0700,
        win_rate=0.6000,
        profit_factor=2.0000,
        n_trades=40,
        avg_trade=130.0,
        equity_curve=sample_backtest_result.equity_curve.copy(),
        drawdown_curve=sample_backtest_result.drawdown_curve.copy(),
        trades=sample_backtest_result.trades.copy(),
        config_hash=sample_backtest_result.config_hash,
        timestamp=sample_backtest_result.timestamp,
    )
    all_results.append((params3, result3))

    return SweepResult(
        best_params=params1,
        best_sharpe=2.5000,
        all_results=all_results,
    )


@pytest.fixture
def sample_close_prices() -> np.ndarray:
    """Create sample close prices array for trades chart testing.

    Returns:
        NumPy array of 365 close prices with realistic random walk.
    """
    np.random.seed(42)
    n_bars = 365
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = base_price * np.cumprod(1 + returns)
    return prices.astype(np.float64)


@pytest.fixture
def minimal_backtest_result() -> BacktestResult:
    """Create a minimal BacktestResult with short equity curve.

    Useful for testing edge cases with minimal data.

    Returns:
        BacktestResult with only 5 bars of data.
    """
    return BacktestResult(
        total_return=0.05,
        sharpe_ratio=1.0,
        sortino_ratio=1.2,
        max_drawdown=0.02,
        win_rate=0.6,
        profit_factor=1.5,
        n_trades=3,
        avg_trade=50.0,
        equity_curve=np.array([100000.0, 101000.0, 100500.0, 102000.0, 105000.0]),
        drawdown_curve=np.array([0.0, 0.0, -0.005, 0.0, 0.0]),
        trades=pd.DataFrame({
            "Entry Time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Exit Time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Entry Price": [100.0, 101.0, 102.0],
            "Exit Price": [101.0, 100.5, 104.0],
            "PnL": [100.0, -50.0, 200.0],
            "Return": [0.01, -0.005, 0.02],
            "Duration": ["1 day"] * 3,
            "Direction": ["Long", "Long", "Long"],
        }),
        config_hash="minimal_hash_123",
        timestamp="2025-01-01T00:00:00+00:00",
    )


@pytest.fixture
def empty_trades_backtest_result() -> BacktestResult:
    """Create a BacktestResult with no trades.

    Useful for testing edge case handling with zero trades.

    Returns:
        BacktestResult with empty trades DataFrame.
    """
    return BacktestResult(
        total_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        n_trades=0,
        avg_trade=0.0,
        equity_curve=np.array([100000.0] * 30),
        drawdown_curve=np.array([0.0] * 30),
        trades=pd.DataFrame(columns=[
            "Entry Time", "Exit Time", "Entry Price", "Exit Price",
            "PnL", "Return", "Duration", "Direction",
        ]),
        config_hash="empty_trades_hash",
        timestamp="2025-01-01T00:00:00+00:00",
    )
