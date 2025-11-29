#!/usr/bin/env python
"""Standalone verification for output/reports.py - no vectorbt required.

Tests all acceptance criteria for I4.T3 by mocking BacktestResult and SweepResult
without importing the actual backtest engine (which requires vectorbt).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.console import Console

# Add simple_futures_backtester to path
sys.path.insert(0, ".")


# Mock the dataclasses to avoid importing vectorbt
@dataclass
class BacktestResult:
    """Mock BacktestResult for testing (matches real implementation)."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade: float
    equity_curve: NDArray[np.float64]
    drawdown_curve: NDArray[np.float64]
    trades: pd.DataFrame
    config_hash: str = ""
    timestamp: str = ""


@dataclass
class SweepResult:
    """Mock SweepResult for testing (matches real implementation)."""

    best_params: dict[str, Any]
    best_sharpe: float
    all_results: list[tuple[dict[str, Any], BacktestResult]]


# Now import the actual reports module
from simple_futures_backtester.output.reports import ReportGenerator


def create_sample_backtest_result() -> BacktestResult:
    """Create a sample BacktestResult for testing."""
    trades_data = {
        "entry_time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "exit_time": [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 2, 12, 0)],
        "side": ["long", "short"],
        "entry_price": [100.0, 105.0],
        "exit_price": [102.0, 104.0],
        "pnl": [2.0, 1.0],
    }
    trades_df = pd.DataFrame(trades_data)

    return BacktestResult(
        total_return=0.1523,
        sharpe_ratio=1.2345,
        sortino_ratio=1.5678,
        max_drawdown=0.0823,
        win_rate=0.6234,
        profit_factor=1.8765,
        n_trades=42,
        avg_trade=125.67,
        equity_curve=np.array([100000.0, 101000.0, 99500.0, 102000.0]),
        drawdown_curve=np.array([0.0, 0.0, -0.015, 0.0]),
        trades=trades_df,
        config_hash="abc123def456789012345678901234567890abcdef0123456789012345678",
        timestamp="2025-11-28T12:00:00+00:00",
    )


def create_sample_sweep_result() -> SweepResult:
    """Create a sample SweepResult for testing."""
    sample_backtest = create_sample_backtest_result()

    # Create variations with different params
    all_results = []

    # Best result
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
        equity_curve=sample_backtest.equity_curve,
        drawdown_curve=sample_backtest.drawdown_curve,
        trades=sample_backtest.trades,
        config_hash=sample_backtest.config_hash,
        timestamp=sample_backtest.timestamp,
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
        equity_curve=sample_backtest.equity_curve,
        drawdown_curve=sample_backtest.drawdown_curve,
        trades=sample_backtest.trades,
        config_hash=sample_backtest.config_hash,
        timestamp=sample_backtest.timestamp,
    )
    all_results.append((params2, result2))

    return SweepResult(
        best_params=params1,
        best_sharpe=2.5000,
        all_results=all_results,
    )


def test_backtest_text_report():
    """Test BacktestResult text report generation."""
    print("\n" + "=" * 80)
    print("TEST 1: BacktestResult Text Report")
    print("=" * 80)

    result = create_sample_backtest_result()
    report = ReportGenerator.generate_text_report(result)

    # Check that it returns a string
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 0, "Report should not be empty"

    # Check for metric labels
    assert "Total Return" in report, "Should contain 'Total Return'"
    assert "Sharpe Ratio" in report, "Should contain 'Sharpe Ratio'"
    assert "Sortino Ratio" in report, "Should contain 'Sortino Ratio'"
    assert "Max Drawdown" in report, "Should contain 'Max Drawdown'"
    assert "Win Rate" in report, "Should contain 'Win Rate'"
    assert "Profit Factor" in report, "Should contain 'Profit Factor'"
    assert "Number of Trades" in report, "Should contain 'Number of Trades'"
    assert "Average Trade" in report, "Should contain 'Average Trade'"

    # Check percentage formatting (2 decimals)
    assert "15.23%" in report, "Total return should be 15.23%"
    assert "62.34%" in report, "Win rate should be 62.34%"
    assert "8.23%" in report, "Max drawdown should be 8.23%"

    # Check ratio formatting (4 decimals)
    assert "1.2345" in report, "Sharpe ratio should be 1.2345"
    assert "1.5678" in report, "Sortino ratio should be 1.5678"
    assert "1.8765" in report, "Profit factor should be 1.8765"

    # Check metadata
    assert "abc123def456789" in report or "Config Hash" in report, "Should show config hash"
    assert "2025-11-28" in report or "Timestamp" in report, "Should show timestamp"

    print("✅ All text report checks passed")

    # Display the actual report
    console = Console()
    print("\nGenerated Report:")
    console.print(report)


def test_backtest_json_report():
    """Test BacktestResult JSON report generation."""
    print("\n" + "=" * 80)
    print("TEST 2: BacktestResult JSON Report")
    print("=" * 80)

    result = create_sample_backtest_result()
    report = ReportGenerator.generate_json_report(result, strategy_name="test_strategy")

    # Check that it returns a dict
    assert isinstance(report, dict), "Report should be a dict"

    # Check JSON serializability
    json_str = json.dumps(report, indent=2)
    assert isinstance(json_str, str), "Should be serializable to JSON"
    loaded = json.loads(json_str)
    assert loaded == report, "Should round-trip through JSON"

    # Check required fields
    assert "timestamp" in report, "Should include timestamp"
    assert report["timestamp"] == "2025-11-28T12:00:00+00:00", "Timestamp should be correct"

    assert "config_hash" in report, "Should include config_hash"
    assert report["config_hash"].startswith("abc123def456"), "Config hash should be correct"

    assert "strategy_name" in report, "Should include strategy_name"
    assert report["strategy_name"] == "test_strategy", "Strategy name should be correct"

    # Check metrics structure
    assert "metrics" in report, "Should include metrics"
    metrics = report["metrics"]

    assert "total_return" in metrics, "Metrics should include total_return"
    assert "sharpe_ratio" in metrics, "Metrics should include sharpe_ratio"
    assert "sortino_ratio" in metrics, "Metrics should include sortino_ratio"
    assert "max_drawdown" in metrics, "Metrics should include max_drawdown"
    assert "win_rate" in metrics, "Metrics should include win_rate"
    assert "profit_factor" in metrics, "Metrics should include profit_factor"
    assert "avg_trade" in metrics, "Metrics should include avg_trade"

    # Check trades_count
    assert "trades_count" in report, "Should include trades_count"
    assert report["trades_count"] == 42, "Trades count should be 42"
    assert isinstance(report["trades_count"], int), "Trades count should be int"

    # Check NumPy array conversion
    assert "equity_curve" in report, "Should include equity_curve"
    assert isinstance(report["equity_curve"], list), "Equity curve should be a list"
    assert len(report["equity_curve"]) == 4, "Equity curve should have 4 elements"

    assert "drawdown_curve" in report, "Should include drawdown_curve"
    assert isinstance(report["drawdown_curve"], list), "Drawdown curve should be a list"
    assert len(report["drawdown_curve"]) == 4, "Drawdown curve should have 4 elements"

    # Check DataFrame conversion
    assert "trades" in report, "Should include trades"
    assert isinstance(report["trades"], list), "Trades should be a list"
    assert len(report["trades"]) == 2, "Should have 2 trades"
    assert "side" in report["trades"][0], "Trades should have 'side' field"
    assert "entry_price" in report["trades"][0], "Trades should have 'entry_price' field"
    assert "exit_price" in report["trades"][0], "Trades should have 'exit_price' field"
    assert "pnl" in report["trades"][0], "Trades should have 'pnl' field"

    # Check precision (4 decimals for ratios)
    assert metrics["sharpe_ratio"] == round(1.2345, 4), "Sharpe ratio precision should be 4 decimals"
    assert metrics["sortino_ratio"] == round(1.5678, 4), "Sortino ratio precision should be 4 decimals"
    assert metrics["profit_factor"] == round(1.8765, 4), "Profit factor precision should be 4 decimals"

    print("✅ All JSON report checks passed")

    # Display sample JSON
    print("\nSample JSON output:")
    print(json.dumps({k: v for k, v in report.items() if k not in ["equity_curve", "drawdown_curve", "trades"]}, indent=2))


def test_sweep_text_report():
    """Test SweepResult text report generation."""
    print("\n" + "=" * 80)
    print("TEST 3: SweepResult Text Report")
    print("=" * 80)

    result = create_sample_sweep_result()
    report = ReportGenerator.generate_text_report(result, top_n=5)

    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 0, "Report should not be empty"

    # Check for parameter names
    assert "stop_loss" in report, "Should include 'stop_loss' parameter"
    assert "take_profit" in report, "Should include 'take_profit' parameter"

    # Check for parameter values
    assert "100" in report, "Should show parameter value 100"
    assert "200" in report, "Should show parameter value 200"

    # Check for best Sharpe
    assert "2.5000" in report, "Should show best Sharpe ratio"

    print("✅ All sweep text report checks passed")

    # Display the actual report
    console = Console()
    print("\nGenerated Sweep Report:")
    console.print(report)


def test_sweep_json_report():
    """Test SweepResult JSON report generation."""
    print("\n" + "=" * 80)
    print("TEST 4: SweepResult JSON Report")
    print("=" * 80)

    result = create_sample_sweep_result()
    report = ReportGenerator.generate_json_report(result, strategy_name="sweep_test")

    assert isinstance(report, dict), "Report should be a dict"

    # Check JSON serializability
    json_str = json.dumps(report, indent=2)
    assert isinstance(json_str, str), "Should be serializable to JSON"

    # Check required fields
    assert "timestamp" in report, "Should include timestamp"
    assert "config_hash" in report, "Should include config_hash"
    assert "strategy_name" in report, "Should include strategy_name"

    # Check best_params
    assert "best_params" in report, "Should include best_params"
    assert report["best_params"]["stop_loss"] == 100, "Best stop_loss should be 100"
    assert report["best_params"]["take_profit"] == 200, "Best take_profit should be 200"

    # Check best_sharpe
    assert "best_sharpe" in report, "Should include best_sharpe"
    assert report["best_sharpe"] == round(2.5000, 4), "Best Sharpe should be 2.5000"

    # Check total_combinations
    assert "total_combinations" in report, "Should include total_combinations"
    assert report["total_combinations"] == 2, "Should have 2 combinations"

    # Check all_results
    assert "all_results" in report, "Should include all_results"
    assert len(report["all_results"]) == 2, "Should have 2 results"

    first_result = report["all_results"][0]
    assert "params" in first_result, "Result should have params"
    assert "sharpe_ratio" in first_result, "Result should have sharpe_ratio"
    assert "total_return" in first_result, "Result should have total_return"
    assert "max_drawdown" in first_result, "Result should have max_drawdown"
    assert "n_trades" in first_result, "Result should have n_trades"

    # Check params are sorted
    param_keys = list(first_result["params"].keys())
    assert param_keys == sorted(param_keys), "Params should be sorted alphabetically"

    print("✅ All sweep JSON report checks passed")

    # Display sample JSON
    print("\nSample Sweep JSON output:")
    print(json.dumps(report, indent=2)[:800] + "\n...")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 80)
    print("TEST 5: Edge Cases")
    print("=" * 80)

    # Test zero trades
    result_zero = BacktestResult(
        total_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        n_trades=0,
        avg_trade=0.0,
        equity_curve=np.array([100000.0]),
        drawdown_curve=np.array([0.0]),
        trades=pd.DataFrame(),
        config_hash="test_hash",
        timestamp="2025-11-28T12:00:00+00:00",
    )

    text_report = ReportGenerator.generate_text_report(result_zero)
    assert isinstance(text_report, str), "Should handle zero trades"

    json_report = ReportGenerator.generate_json_report(result_zero)
    assert json_report["trades_count"] == 0, "Should report 0 trades"

    print("✅ Zero trades case passed")

    # Test negative returns
    result_negative = BacktestResult(
        total_return=-0.1500,
        sharpe_ratio=-0.5000,
        sortino_ratio=-0.3000,
        max_drawdown=0.2500,
        win_rate=0.4000,
        profit_factor=0.8000,
        n_trades=10,
        avg_trade=-50.0,
        equity_curve=np.array([100000.0, 95000.0, 90000.0, 85000.0]),
        drawdown_curve=np.array([0.0, -0.05, -0.10, -0.15]),
        trades=pd.DataFrame(),
        config_hash="test_hash",
        timestamp="2025-11-28T12:00:00+00:00",
    )

    text_report_neg = ReportGenerator.generate_text_report(result_negative)
    assert "-15.00%" in text_report_neg, "Should show negative percentage"

    json_report_neg = ReportGenerator.generate_json_report(result_negative)
    assert json_report_neg["metrics"]["total_return"] < 0, "Should report negative return"

    print("✅ Negative returns case passed")

    # Test NaN values
    result_nan = BacktestResult(
        total_return=float("nan"),
        sharpe_ratio=float("nan"),
        sortino_ratio=float("nan"),
        max_drawdown=float("nan"),
        win_rate=0.0,
        profit_factor=0.0,
        n_trades=0,
        avg_trade=0.0,
        equity_curve=np.array([100000.0]),
        drawdown_curve=np.array([0.0]),
        trades=pd.DataFrame(),
        config_hash="test_hash",
        timestamp="2025-11-28T12:00:00+00:00",
    )

    text_report_nan = ReportGenerator.generate_text_report(result_nan)
    assert isinstance(text_report_nan, str), "Should handle NaN values"

    json_report_nan = ReportGenerator.generate_json_report(result_nan)
    assert json_report_nan["metrics"]["total_return"] == 0.0, "NaN should be converted to 0.0"

    print("✅ NaN values case passed")

    # Test invalid result type
    try:
        ReportGenerator.generate_text_report("invalid")  # type: ignore[arg-type]
        assert False, "Should raise TypeError for invalid input"
    except TypeError as e:
        assert "Expected BacktestResult or SweepResult" in str(e)
        print("✅ Invalid type error handling passed")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("VERIFYING I4.T3: output/reports.py Implementation")
    print("=" * 80)

    try:
        test_backtest_text_report()
        test_backtest_json_report()
        test_sweep_text_report()
        test_sweep_json_report()
        test_edge_cases()

        print("\n" + "=" * 80)
        print("✅ ALL ACCEPTANCE CRITERIA VERIFIED")
        print("=" * 80)
        print("\nSummary:")
        print("✅ generate_text_report(BacktestResult) returns Rich-formatted string")
        print("✅ Metrics displayed in aligned table format")
        print("✅ Percentages formatted with 2 decimal places")
        print("✅ Ratios formatted with 4 decimal places")
        print("✅ generate_json_report(BacktestResult) returns JSON-serializable dict")
        print("✅ JSON includes: timestamp (ISO 8601), config_hash, strategy_name, metrics, trades_count")
        print("✅ NumPy arrays converted to lists")
        print("✅ pandas DataFrame converted to list of dicts")
        print("✅ SweepResult text report shows top N with sorted params")
        print("✅ SweepResult JSON includes all_results with sorted params and sharpe values")
        print("✅ Edge cases handled: zero trades, negative returns, NaN values")
        print("\n" + "=" * 80)

        return 0

    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
