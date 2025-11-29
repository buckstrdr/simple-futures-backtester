#!/usr/bin/env python
"""Verification script for I4.T3 - Output Reports Module.

Tests that reports.py meets all acceptance criteria:
1. generate_text_report(result) returns string with Rich markup
2. Metrics displayed in aligned table format
3. generate_json_report(result) returns dict serializable via json.dumps()
4. JSON includes: timestamp (ISO 8601), config_hash, strategy_name, metrics, trades_count
5. For SweepResult: includes all_results with sorted params and sharpe values
6. Precision: returns to 4 decimals, percentages to 2 decimals
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from simple_futures_backtester.backtest.engine import BacktestResult
from simple_futures_backtester.backtest.sweep import SweepResult
from simple_futures_backtester.output.reports import ReportGenerator


def test_backtest_text_report() -> None:
    """Test generate_text_report for BacktestResult."""
    print("\n[TEST 1] BacktestResult Text Report")
    print("=" * 60)

    # Create sample BacktestResult
    result = BacktestResult(
        total_return=0.1523,
        sharpe_ratio=1.2345,
        sortino_ratio=1.5678,
        max_drawdown=0.0823,
        win_rate=0.6234,
        profit_factor=1.8765,
        n_trades=42,
        avg_trade=125.6789,
        equity_curve=np.array([100000.0, 101000.0, 99500.0, 102000.0]),
        drawdown_curve=np.array([0.0, 0.0, -0.015, 0.0]),
        trades=pd.DataFrame({"entry_price": [100, 101], "exit_price": [102, 100]}),
        config_hash="abc123def456",
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    # Generate text report
    text_report = ReportGenerator.generate_text_report(result)

    # Verify it's a string
    assert isinstance(text_report, str), "Text report must be a string"
    print(f"✓ Text report is a string ({len(text_report)} chars)")

    # Verify it contains formatted table (box drawing characters or ANSI)
    has_formatting = (
        "┃" in text_report or  # Box drawing
        "│" in text_report or  # Box drawing
        "\x1b[" in text_report  # ANSI codes
    )
    assert has_formatting, "Must contain Rich formatting (table or ANSI codes)"
    print("✓ Contains Rich formatting (rendered table)")

    # Verify key metrics are present
    assert "Total Return" in text_report, "Must include Total Return"
    assert "Sharpe Ratio" in text_report, "Must include Sharpe Ratio"
    assert "15.23%" in text_report, "Total return must be formatted as percentage (2 decimals)"
    assert "1.2345" in text_report, "Sharpe ratio must have 4 decimals"
    print("✓ Contains all key metrics with correct precision")

    print("\nSample output:")
    print(text_report[:500] + "...")


def test_backtest_json_report() -> None:
    """Test generate_json_report for BacktestResult."""
    print("\n[TEST 2] BacktestResult JSON Report")
    print("=" * 60)

    # Create sample BacktestResult
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    result = BacktestResult(
        total_return=0.1523,
        sharpe_ratio=1.2345,
        sortino_ratio=1.5678,
        max_drawdown=0.0823,
        win_rate=0.6234,
        profit_factor=1.8765,
        n_trades=42,
        avg_trade=125.6789,
        equity_curve=np.array([100000.0, 101000.0, 99500.0, 102000.0]),
        drawdown_curve=np.array([0.0, 0.0, -0.015, 0.0]),
        trades=pd.DataFrame({"entry_price": [100, 101], "exit_price": [102, 100]}),
        config_hash="abc123def456",
        timestamp=timestamp,
    )

    # Generate JSON report
    json_report = ReportGenerator.generate_json_report(result, strategy_name="momentum")

    # Verify it's a dict
    assert isinstance(json_report, dict), "JSON report must be a dict"
    print(f"✓ JSON report is a dict with {len(json_report)} keys")

    # Verify it's JSON-serializable
    try:
        json_str = json.dumps(json_report, indent=2)
        print(f"✓ JSON report is serializable ({len(json_str)} bytes)")
    except Exception as e:
        raise AssertionError(f"JSON report must be serializable: {e}")

    # Verify required fields
    assert "timestamp" in json_report, "Must include timestamp"
    assert json_report["timestamp"] == timestamp, "Timestamp must match"
    print(f"✓ Includes timestamp (ISO 8601): {json_report['timestamp'][:26]}...")

    assert "config_hash" in json_report, "Must include config_hash"
    assert json_report["config_hash"] == "abc123def456", "Config hash must match"
    print(f"✓ Includes config_hash: {json_report['config_hash']}")

    assert "strategy_name" in json_report, "Must include strategy_name"
    assert json_report["strategy_name"] == "momentum", "Strategy name must match"
    print(f"✓ Includes strategy_name: {json_report['strategy_name']}")

    assert "metrics" in json_report, "Must include metrics"
    assert isinstance(json_report["metrics"], dict), "Metrics must be a dict"
    print(f"✓ Includes metrics dict with {len(json_report['metrics'])} entries")

    assert "trades_count" in json_report, "Must include trades_count"
    assert json_report["trades_count"] == 42, "Trades count must match"
    print(f"✓ Includes trades_count: {json_report['trades_count']}")

    # Verify metrics structure
    metrics = json_report["metrics"]
    required_metrics = [
        "total_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade",
    ]
    for metric in required_metrics:
        assert metric in metrics, f"Metrics must include {metric}"
    print(f"✓ All required metrics present: {', '.join(required_metrics)}")

    # Verify precision (values should be rounded to 4 decimals)
    assert metrics["sharpe_ratio"] == 1.2345, "Sharpe ratio precision"
    assert metrics["total_return"] == 0.1523, "Total return precision"
    print("✓ Metric precision correct (4 decimals)")

    print(f"\nSample JSON structure:")
    print(json.dumps(json_report, indent=2)[:800] + "...")


def test_sweep_text_report() -> None:
    """Test generate_text_report for SweepResult."""
    print("\n[TEST 3] SweepResult Text Report")
    print("=" * 60)

    # Create sample SweepResult
    backtest1 = BacktestResult(
        total_return=0.25,
        sharpe_ratio=2.1,
        sortino_ratio=2.5,
        max_drawdown=0.1,
        win_rate=0.65,
        profit_factor=2.2,
        n_trades=50,
        avg_trade=150.0,
        equity_curve=np.array([100000.0, 125000.0]),
        drawdown_curve=np.array([0.0, -0.05]),
        trades=pd.DataFrame(),
        config_hash="sweep1",
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    backtest2 = BacktestResult(
        total_return=0.18,
        sharpe_ratio=1.8,
        sortino_ratio=2.0,
        max_drawdown=0.08,
        win_rate=0.60,
        profit_factor=1.9,
        n_trades=45,
        avg_trade=120.0,
        equity_curve=np.array([100000.0, 118000.0]),
        drawdown_curve=np.array([0.0, -0.03]),
        trades=pd.DataFrame(),
        config_hash="sweep2",
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    sweep_result = SweepResult(
        best_params={"ma_period": 20, "threshold": 0.02},
        best_sharpe=2.1,
        all_results=[
            ({"ma_period": 20, "threshold": 0.02}, backtest1),
            ({"ma_period": 15, "threshold": 0.01}, backtest2),
        ],
    )

    # Generate text report
    text_report = ReportGenerator.generate_text_report(sweep_result, top_n=5)

    # Verify it's a string
    assert isinstance(text_report, str), "Text report must be a string"
    print(f"✓ Text report is a string ({len(text_report)} chars)")

    # Verify it contains sweep-specific content
    assert "Sweep" in text_report or "Parameter" in text_report, "Must indicate sweep results"
    assert "2.1000" in text_report, "Best sharpe must appear with 4 decimals"
    print("✓ Contains sweep-specific information")

    print("\nSample output:")
    print(text_report[:500] + "...")


def test_sweep_json_report() -> None:
    """Test generate_json_report for SweepResult."""
    print("\n[TEST 4] SweepResult JSON Report")
    print("=" * 60)

    # Create sample SweepResult
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    backtest1 = BacktestResult(
        total_return=0.25,
        sharpe_ratio=2.1,
        sortino_ratio=2.5,
        max_drawdown=0.1,
        win_rate=0.65,
        profit_factor=2.2,
        n_trades=50,
        avg_trade=150.0,
        equity_curve=np.array([100000.0, 125000.0]),
        drawdown_curve=np.array([0.0, -0.05]),
        trades=pd.DataFrame(),
        config_hash="sweep1",
        timestamp=timestamp,
    )

    backtest2 = BacktestResult(
        total_return=0.18,
        sharpe_ratio=1.8,
        sortino_ratio=2.0,
        max_drawdown=0.08,
        win_rate=0.60,
        profit_factor=1.9,
        n_trades=45,
        avg_trade=120.0,
        equity_curve=np.array([100000.0, 118000.0]),
        drawdown_curve=np.array([0.0, -0.03]),
        trades=pd.DataFrame(),
        config_hash="sweep2",
        timestamp=timestamp,
    )

    sweep_result = SweepResult(
        best_params={"threshold": 0.02, "ma_period": 20},  # Unsorted
        best_sharpe=2.1,
        all_results=[
            ({"threshold": 0.02, "ma_period": 20}, backtest1),  # Unsorted
            ({"threshold": 0.01, "ma_period": 15}, backtest2),  # Unsorted
        ],
    )

    # Generate JSON report
    json_report = ReportGenerator.generate_json_report(sweep_result, strategy_name="sweep_test")

    # Verify it's a dict
    assert isinstance(json_report, dict), "JSON report must be a dict"
    print(f"✓ JSON report is a dict with {len(json_report)} keys")

    # Verify it's JSON-serializable
    try:
        json_str = json.dumps(json_report, indent=2)
        print(f"✓ JSON report is serializable ({len(json_str)} bytes)")
    except Exception as e:
        raise AssertionError(f"JSON report must be serializable: {e}")

    # Verify required fields
    assert "timestamp" in json_report, "Must include timestamp"
    assert "config_hash" in json_report, "Must include config_hash"
    assert "strategy_name" in json_report, "Must include strategy_name"
    assert "best_params" in json_report, "Must include best_params"
    assert "best_sharpe" in json_report, "Must include best_sharpe"
    assert "total_combinations" in json_report, "Must include total_combinations"
    assert "all_results" in json_report, "Must include all_results"
    print("✓ All required fields present")

    # Verify params are sorted
    best_params_keys = list(json_report["best_params"].keys())
    assert best_params_keys == sorted(best_params_keys), "best_params must be sorted by key"
    print(f"✓ best_params sorted: {list(json_report['best_params'].keys())}")

    # Verify all_results structure
    all_results = json_report["all_results"]
    assert len(all_results) == 2, "Must include all results"
    assert all(isinstance(r, dict) for r in all_results), "Each result must be a dict"
    assert all("params" in r for r in all_results), "Each result must have params"
    assert all("sharpe_ratio" in r for r in all_results), "Each result must have sharpe_ratio"
    print(f"✓ all_results contains {len(all_results)} entries with correct structure")

    # Verify params in all_results are sorted
    for idx, result in enumerate(all_results):
        params_keys = list(result["params"].keys())
        assert params_keys == sorted(params_keys), f"Result {idx} params must be sorted"
    print("✓ All params dicts are sorted by key")

    print(f"\nSample JSON structure:")
    print(json.dumps(json_report, indent=2)[:1000] + "...")


def main() -> None:
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("VERIFICATION: I4.T3 - Output Reports Module")
    print("=" * 60)

    try:
        test_backtest_text_report()
        test_backtest_json_report()
        test_sweep_text_report()
        test_sweep_json_report()

        print("\n" + "=" * 60)
        print("✅ ALL ACCEPTANCE CRITERIA MET")
        print("=" * 60)
        print("\nSummary:")
        print("✓ generate_text_report(BacktestResult) returns Rich-formatted string")
        print("✓ generate_text_report(SweepResult) returns Rich-formatted string")
        print("✓ generate_json_report(BacktestResult) returns JSON-serializable dict")
        print("✓ generate_json_report(SweepResult) returns JSON-serializable dict")
        print("✓ JSON includes: timestamp, config_hash, strategy_name, metrics, trades_count")
        print("✓ SweepResult JSON includes all_results with sorted params")
        print("✓ Precision: 4 decimals for ratios, 2 decimals for percentages")

    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
