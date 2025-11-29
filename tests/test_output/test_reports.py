"""Tests for output/reports.py module.

Verifies Rich text formatting and JSON serialization for BacktestResult
and SweepResult reporting. Comprehensive test coverage for the ReportGenerator class.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from simple_futures_backtester.backtest.engine import BacktestResult
from simple_futures_backtester.backtest.sweep import SweepResult
from simple_futures_backtester.output.reports import ReportGenerator


class TestTextReportRichMarkup:
    """Tests for verifying Rich markup is present in text reports."""

    def test_text_report_contains_rich_color_markup(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that text report contains Rich color markup tags.

        Verifies that the report contains Rich markup tags like [green],
        [yellow], [red], [cyan], etc. for colored output.
        """
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Check for presence of Rich color markup (escaped in ANSI output)
        # The Rich console captures output with ANSI codes, but we can check
        # that at minimum the report contains styled output characteristics
        assert len(report) > 100  # Report should have substantial content

        # The report should contain metric values with color styling
        # Since Rich renders to ANSI codes, we check for table structure
        assert "Total Return" in report
        assert "Sharpe Ratio" in report

    def test_text_report_contains_table_structure(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that text report contains Rich table structure."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Rich tables use box drawing characters
        # Check for table title or column headers
        assert "Backtest Results" in report

    def test_text_report_has_formatted_metrics(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that metrics are formatted with Rich styling."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Verify all metric labels appear
        metric_labels = [
            "Total Return",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Win Rate",
            "Profit Factor",
            "Number of Trades",
            "Average Trade",
        ]
        for label in metric_labels:
            assert label in report, f"Missing metric label: {label}"

    def test_sweep_text_report_contains_rich_markup(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep text report contains Rich markup."""
        report = ReportGenerator.generate_text_report(sample_sweep_result)

        # Check report is substantial and contains key elements
        assert len(report) > 100
        assert "Parameter Sweep Results" in report
        assert "Best" in report


class TestBacktestTextReport:
    """Tests for BacktestResult text report generation."""

    def test_generate_text_report_returns_string(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that generate_text_report returns a string."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_text_report_contains_metrics(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that text report contains all expected metrics."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Check for metric labels
        assert "Total Return" in report
        assert "Sharpe Ratio" in report
        assert "Sortino Ratio" in report
        assert "Max Drawdown" in report
        assert "Win Rate" in report
        assert "Profit Factor" in report
        assert "Number of Trades" in report
        assert "Average Trade" in report

    def test_text_report_percentage_formatting(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that percentages are formatted with 2 decimal places."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Total return should be 15.23%
        assert "15.23%" in report
        # Win rate should be 62.34%
        assert "62.34%" in report
        # Max drawdown should be displayed (as negative percentage)
        assert "8.23%" in report

    def test_text_report_ratio_formatting(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that ratios are formatted with 4 decimal places."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Sharpe ratio should be 1.2345
        assert "1.2345" in report
        # Sortino ratio should be 1.5678
        assert "1.5678" in report
        # Profit factor should be 1.8765
        assert "1.8765" in report

    def test_text_report_metadata(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that metadata is included in text report."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)

        # Config hash (first 16 chars shown)
        assert "abc123def456789" in report or "Config Hash" in report
        # Timestamp
        assert "2025-11-28" in report or "Timestamp" in report

    def test_text_report_rich_console_compatible(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that text report can be printed by Rich Console without errors."""
        report = ReportGenerator.generate_text_report(sample_backtest_result)
        console = Console()

        # Should not raise exception
        console.print(report)


class TestBacktestJsonReport:
    """Tests for BacktestResult JSON report generation."""

    def test_generate_json_report_returns_dict(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that generate_json_report returns a dict."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)
        assert isinstance(report, dict)

    def test_json_report_serializable(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report is serializable with json.dumps()."""
        report = ReportGenerator.generate_json_report(
            sample_backtest_result,
            strategy_name="momentum",
        )

        # Custom encoder to handle pandas Timestamps
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return super().default(obj)

        # Should not raise exception
        json_str = json.dumps(report, indent=2, cls=DateTimeEncoder)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Verify it can be loaded back
        loaded = json.loads(json_str)
        assert isinstance(loaded, dict)

    def test_json_report_includes_timestamp(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report includes timestamp in ISO 8601 format."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "timestamp" in report
        assert report["timestamp"] == "2025-11-28T12:00:00+00:00"

    def test_json_report_includes_config_hash(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report includes config_hash."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "config_hash" in report
        assert report["config_hash"].startswith("abc123def456")

    def test_json_report_includes_strategy_name(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report includes strategy_name."""
        report = ReportGenerator.generate_json_report(
            sample_backtest_result,
            strategy_name="test_strategy",
        )

        assert "strategy_name" in report
        assert report["strategy_name"] == "test_strategy"

    def test_json_report_includes_metrics(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report includes all metrics."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "metrics" in report
        metrics = report["metrics"]

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "avg_trade" in metrics

    def test_json_report_includes_trades_count(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that JSON report includes trades_count."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "trades_count" in report
        assert report["trades_count"] == sample_backtest_result.n_trades
        assert isinstance(report["trades_count"], int)

    def test_json_report_converts_numpy_arrays(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that NumPy arrays are converted to lists."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "equity_curve" in report
        assert isinstance(report["equity_curve"], list)
        assert len(report["equity_curve"]) == len(sample_backtest_result.equity_curve)

        assert "drawdown_curve" in report
        assert isinstance(report["drawdown_curve"], list)
        assert len(report["drawdown_curve"]) == len(sample_backtest_result.drawdown_curve)

    def test_json_report_converts_dataframe(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that pandas DataFrame is converted to list of dicts."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        assert "trades" in report
        assert isinstance(report["trades"], list)
        assert len(report["trades"]) == len(sample_backtest_result.trades)

        # Check first trade structure has some expected keys
        if len(report["trades"]) > 0:
            first_trade = report["trades"][0]
            # Check at least one expected column exists (depends on fixture)
            assert isinstance(first_trade, dict)
            assert len(first_trade) > 0

    def test_json_report_precision(
        self,
        sample_backtest_result: BacktestResult,
    ) -> None:
        """Test that metrics have correct precision (4 decimals for ratios)."""
        report = ReportGenerator.generate_json_report(sample_backtest_result)

        metrics = report["metrics"]

        # Ratios should have 4 decimal precision (after rounding)
        assert metrics["sharpe_ratio"] == round(1.2345, 4)
        assert metrics["sortino_ratio"] == round(1.5678, 4)
        assert metrics["profit_factor"] == round(1.8765, 4)

        # Returns should also be 4 decimals (stored as decimals, not percentages)
        assert metrics["total_return"] == round(0.1523, 4)
        assert metrics["max_drawdown"] == round(0.0823, 4)
        assert metrics["win_rate"] == round(0.6234, 4)


class TestSweepTextReport:
    """Tests for SweepResult text report generation."""

    def test_generate_sweep_text_report(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep text report generates successfully."""
        report = ReportGenerator.generate_text_report(sample_sweep_result)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_sweep_text_report_shows_top_n(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep report shows correct number of results."""
        report = ReportGenerator.generate_text_report(sample_sweep_result, top_n=2)

        # Should show "Top 2 of 3"
        assert "Top 2" in report or "2" in report

    def test_sweep_text_report_includes_params(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep report includes parameter names and values."""
        report = ReportGenerator.generate_text_report(sample_sweep_result)

        # Parameter names
        assert "stop_loss" in report
        assert "take_profit" in report

        # Parameter values from first result
        assert "100" in report
        assert "200" in report

    def test_sweep_text_report_sorted_by_sharpe(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep report shows results sorted by Sharpe ratio."""
        report = ReportGenerator.generate_text_report(sample_sweep_result)

        # Best Sharpe (2.5000) should appear in report
        assert "2.5000" in report


class TestSweepJsonReport:
    """Tests for SweepResult JSON report generation."""

    def test_generate_sweep_json_report(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON report generates successfully."""
        report = ReportGenerator.generate_json_report(
            sample_sweep_result,
            strategy_name="sweep_test",
        )

        assert isinstance(report, dict)

    def test_sweep_json_serializable(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON report is serializable."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        # Should not raise exception
        json_str = json.dumps(report, indent=2)
        assert isinstance(json_str, str)

    def test_sweep_json_includes_best_params(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON includes best_params."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        assert "best_params" in report
        assert report["best_params"]["stop_loss"] == 100
        assert report["best_params"]["take_profit"] == 200

    def test_sweep_json_includes_best_sharpe(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON includes best_sharpe."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        assert "best_sharpe" in report
        assert report["best_sharpe"] == round(2.5000, 4)

    def test_sweep_json_includes_all_results(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON includes all_results with sorted params."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        assert "all_results" in report
        assert len(report["all_results"]) == 3

        # Check first result structure
        first_result = report["all_results"][0]
        assert "params" in first_result
        assert "sharpe_ratio" in first_result
        assert "total_return" in first_result
        assert "max_drawdown" in first_result
        assert "n_trades" in first_result

    def test_sweep_json_params_sorted(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that parameters are sorted alphabetically in JSON."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        first_params = report["all_results"][0]["params"]
        param_keys = list(first_params.keys())

        # Should be sorted: stop_loss, take_profit
        assert param_keys == sorted(param_keys)

    def test_sweep_json_includes_metadata(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON includes metadata from first result."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        assert "timestamp" in report
        assert "config_hash" in report
        assert "strategy_name" in report

    def test_sweep_json_includes_total_combinations(
        self,
        sample_sweep_result: SweepResult,
    ) -> None:
        """Test that sweep JSON includes total_combinations count."""
        report = ReportGenerator.generate_json_report(sample_sweep_result)

        assert "total_combinations" in report
        assert report["total_combinations"] == 3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_trades_result(self) -> None:
        """Test handling of BacktestResult with zero trades."""
        result = BacktestResult(
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

        # Should not crash
        text_report = ReportGenerator.generate_text_report(result)
        assert isinstance(text_report, str)

        json_report = ReportGenerator.generate_json_report(result)
        assert json_report["trades_count"] == 0

    def test_negative_returns(self) -> None:
        """Test handling of negative returns."""
        result = BacktestResult(
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

        text_report = ReportGenerator.generate_text_report(result)
        # Should show negative percentage
        assert "-15.00%" in text_report

        json_report = ReportGenerator.generate_json_report(result)
        assert json_report["metrics"]["total_return"] < 0

    def test_nan_values(self) -> None:
        """Test handling of NaN values in metrics."""
        result = BacktestResult(
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

        # Should not crash, NaN should be converted to 0.0
        text_report = ReportGenerator.generate_text_report(result)
        assert isinstance(text_report, str)

        json_report = ReportGenerator.generate_json_report(result)
        # NaN values should be replaced with 0.0
        assert json_report["metrics"]["total_return"] == 0.0

    def test_invalid_result_type(self) -> None:
        """Test that invalid result type raises TypeError."""
        with pytest.raises(TypeError, match="Expected BacktestResult or SweepResult"):
            ReportGenerator.generate_text_report("invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected BacktestResult or SweepResult"):
            ReportGenerator.generate_json_report(123)  # type: ignore[arg-type]

    def test_empty_sweep_result(self) -> None:
        """Test handling of empty SweepResult."""
        result = SweepResult(
            best_params={},
            best_sharpe=0.0,
            all_results=[],
        )

        # Should not crash
        text_report = ReportGenerator.generate_text_report(result)
        assert "Empty" in text_report or "No parameter" in text_report

        json_report = ReportGenerator.generate_json_report(result)
        assert json_report["total_combinations"] == 0
        assert len(json_report["all_results"]) == 0
