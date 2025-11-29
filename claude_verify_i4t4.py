#!/usr/bin/env python3
"""Verification script for I4.T4 - Chart Generation Module.

Tests ChartFactory implementation against all acceptance criteria.
"""

import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simple_futures_backtester.backtest.engine import BacktestResult
from simple_futures_backtester.output.charts import ChartFactory


@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    passed: bool
    message: str


def create_sample_result() -> BacktestResult:
    """Create sample BacktestResult for testing."""
    n_bars = 100
    equity_curve = np.linspace(100000, 115000, n_bars) + np.random.randn(n_bars) * 500

    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown_curve = (equity_curve - peak) / peak

    # Create sample trades
    trades = pd.DataFrame({
        'Entry Time': pd.date_range('2020-01-01', periods=10, freq='D'),
        'Exit Time': pd.date_range('2020-01-02', periods=10, freq='D'),
        'Entry Price': np.random.uniform(100, 110, 10),
        'Exit Price': np.random.uniform(100, 110, 10),
        'PnL': np.random.uniform(-500, 1000, 10),
        'Return': np.random.uniform(-0.05, 0.1, 10),
        'Duration': np.random.randint(1, 10, 10),
        'Direction': ['Long', 'Short'] * 5,
    })

    return BacktestResult(
        total_return=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.10,
        win_rate=0.60,
        profit_factor=1.8,
        n_trades=10,
        avg_trade=500.0,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trades=trades,
        config_hash="test123",
        timestamp="2025-11-28T12:00:00Z",
    )


def verify_equity_curve() -> List[TestResult]:
    """Verify create_equity_curve() implementation."""
    results = []
    result = create_sample_result()

    # Test 1: Basic functionality
    try:
        fig = ChartFactory.create_equity_curve(result)
        results.append(TestResult(
            "create_equity_curve returns go.Figure",
            isinstance(fig, go.Figure),
            f"Expected go.Figure, got {type(fig)}"
        ))
    except Exception as e:
        results.append(TestResult(
            "create_equity_curve returns go.Figure",
            False,
            f"Exception: {e}"
        ))
        return results

    # Test 2: Has equity line trace
    has_equity = any('Equity' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Equity curve has equity line trace",
        has_equity,
        "Equity trace found" if has_equity else "No equity trace found"
    ))

    # Test 3: Has benchmark line
    has_benchmark = any('Benchmark' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Equity curve has benchmark line",
        has_benchmark,
        "Benchmark trace found" if has_benchmark else "No benchmark trace found"
    ))

    # Test 4: Has title
    has_title = fig.layout.title is not None and 'Equity' in str(fig.layout.title.text)
    results.append(TestResult(
        "Equity curve has title",
        has_title,
        f"Title: {fig.layout.title.text if fig.layout.title else 'None'}"
    ))

    # Test 5: Has axis labels
    has_xaxis = fig.layout.xaxis.title is not None
    has_yaxis = fig.layout.yaxis.title is not None
    results.append(TestResult(
        "Equity curve has axis labels",
        has_xaxis and has_yaxis,
        f"X-axis: {fig.layout.xaxis.title.text if has_xaxis else 'None'}, Y-axis: {fig.layout.yaxis.title.text if has_yaxis else 'None'}"
    ))

    # Test 6: Has legend
    has_legend = fig.layout.showlegend is not False
    results.append(TestResult(
        "Equity curve has legend",
        has_legend,
        "Legend enabled" if has_legend else "Legend disabled"
    ))

    # Test 7: Dark theme works
    try:
        fig_dark = ChartFactory.create_equity_curve(result, dark_theme=True)
        is_dark = 'dark' in str(fig_dark.layout.template).lower()
        results.append(TestResult(
            "Equity curve dark_theme parameter works",
            is_dark,
            f"Template: {fig_dark.layout.template}"
        ))
    except Exception as e:
        results.append(TestResult(
            "Equity curve dark_theme parameter works",
            False,
            f"Exception: {e}"
        ))

    # Test 8: Uses correct color (#4FC3F7)
    equity_trace = next((trace for trace in fig.data if 'Equity' in str(trace.name)), None)
    if equity_trace:
        color_correct = '#4FC3F7' in str(equity_trace.line.color).upper()
        results.append(TestResult(
            "Equity curve uses COLOR_PRIMARY (#4FC3F7)",
            color_correct,
            f"Color: {equity_trace.line.color}"
        ))

    return results


def verify_drawdown_chart() -> List[TestResult]:
    """Verify create_drawdown_chart() implementation."""
    results = []
    result = create_sample_result()

    # Test 1: Basic functionality
    try:
        fig = ChartFactory.create_drawdown_chart(result)
        results.append(TestResult(
            "create_drawdown_chart returns go.Figure",
            isinstance(fig, go.Figure),
            f"Expected go.Figure, got {type(fig)}"
        ))
    except Exception as e:
        results.append(TestResult(
            "create_drawdown_chart returns go.Figure",
            False,
            f"Exception: {e}"
        ))
        return results

    # Test 2: Has drawdown area
    has_drawdown = any('Drawdown' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Drawdown chart has drawdown trace",
        has_drawdown,
        "Drawdown trace found" if has_drawdown else "No drawdown trace found"
    ))

    # Test 3: Has fill (area chart)
    dd_trace = next((trace for trace in fig.data if 'Drawdown' in str(trace.name)), None)
    if dd_trace:
        has_fill = dd_trace.fill == 'tozeroy'
        results.append(TestResult(
            "Drawdown chart uses area fill (tozeroy)",
            has_fill,
            f"Fill: {dd_trace.fill}"
        ))

    # Test 4: Has title
    has_title = fig.layout.title is not None and 'Drawdown' in str(fig.layout.title.text)
    results.append(TestResult(
        "Drawdown chart has title",
        has_title,
        f"Title: {fig.layout.title.text if fig.layout.title else 'None'}"
    ))

    # Test 5: Has axis labels
    has_xaxis = fig.layout.xaxis.title is not None
    has_yaxis = fig.layout.yaxis.title is not None and '%' in str(fig.layout.yaxis.title.text)
    results.append(TestResult(
        "Drawdown chart has axis labels with %",
        has_xaxis and has_yaxis,
        f"X-axis: {fig.layout.xaxis.title.text if has_xaxis else 'None'}, Y-axis: {fig.layout.yaxis.title.text if has_yaxis else 'None'}"
    ))

    # Test 6: Y-axis has % suffix
    has_pct_suffix = '%' in str(fig.layout.yaxis.ticksuffix)
    results.append(TestResult(
        "Drawdown chart y-axis has % suffix",
        has_pct_suffix,
        f"Tick suffix: {fig.layout.yaxis.ticksuffix}"
    ))

    # Test 7: Uses correct color (#E53935)
    if dd_trace:
        color_correct = '#E53935' in str(dd_trace.line.color).upper()
        results.append(TestResult(
            "Drawdown chart uses COLOR_DANGER (#E53935)",
            color_correct,
            f"Color: {dd_trace.line.color}"
        ))

    return results


def verify_trades_chart() -> List[TestResult]:
    """Verify create_trades_chart() implementation."""
    results = []
    result = create_sample_result()
    close_prices = np.linspace(100, 110, len(result.equity_curve))

    # Test 1: Basic functionality
    try:
        fig = ChartFactory.create_trades_chart(result, close_prices)
        results.append(TestResult(
            "create_trades_chart returns go.Figure",
            isinstance(fig, go.Figure),
            f"Expected go.Figure, got {type(fig)}"
        ))
    except Exception as e:
        results.append(TestResult(
            "create_trades_chart returns go.Figure",
            False,
            f"Exception: {e}"
        ))
        return results

    # Test 2: Has price trace (line or candlestick)
    has_price = any('Price' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Trades chart has price trace",
        has_price,
        "Price trace found" if has_price else "No price trace found"
    ))

    # Test 3: Has entry markers
    has_entry = any('Entry' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Trades chart has entry markers",
        has_entry,
        "Entry markers found" if has_entry else "No entry markers found"
    ))

    # Test 4: Has exit markers
    has_exit = any('Exit' in str(trace.name) for trace in fig.data)
    results.append(TestResult(
        "Trades chart has exit markers",
        has_exit,
        "Exit markers found" if has_exit else "No exit markers found"
    ))

    # Test 5: Entry markers are triangle-up
    entry_trace = next((trace for trace in fig.data if 'Entry' in str(trace.name)), None)
    if entry_trace:
        is_triangle_up = 'triangle-up' in str(entry_trace.marker.symbol)
        results.append(TestResult(
            "Entry markers use triangle-up symbol",
            is_triangle_up,
            f"Symbol: {entry_trace.marker.symbol}"
        ))

    # Test 6: Exit markers are triangle-down
    exit_trace = next((trace for trace in fig.data if 'Exit' in str(trace.name)), None)
    if exit_trace:
        is_triangle_down = 'triangle-down' in str(exit_trace.marker.symbol)
        results.append(TestResult(
            "Exit markers use triangle-down symbol",
            is_triangle_down,
            f"Symbol: {exit_trace.marker.symbol}"
        ))

    # Test 7: Has title
    has_title = fig.layout.title is not None and 'Trades' in str(fig.layout.title.text)
    results.append(TestResult(
        "Trades chart has title",
        has_title,
        f"Title: {fig.layout.title.text if fig.layout.title else 'None'}"
    ))

    # Test 8: Has axis labels
    has_xaxis = fig.layout.xaxis.title is not None
    has_yaxis = fig.layout.yaxis.title is not None
    results.append(TestResult(
        "Trades chart has axis labels",
        has_xaxis and has_yaxis,
        f"X-axis: {fig.layout.xaxis.title.text if has_xaxis else 'None'}, Y-axis: {fig.layout.yaxis.title.text if has_yaxis else 'None'}"
    ))

    # Test 9: Range slider hidden
    slider_hidden = fig.layout.xaxis.rangeslider.visible is False
    results.append(TestResult(
        "Trades chart hides range slider",
        slider_hidden,
        f"Rangeslider visible: {fig.layout.xaxis.rangeslider.visible}"
    ))

    # Test 10: Works with empty trades
    try:
        empty_result = create_sample_result()
        empty_result.trades = pd.DataFrame()
        fig_empty = ChartFactory.create_trades_chart(empty_result, close_prices)
        results.append(TestResult(
            "Trades chart handles empty trades DataFrame",
            isinstance(fig_empty, go.Figure),
            "No exception with empty trades"
        ))
    except Exception as e:
        results.append(TestResult(
            "Trades chart handles empty trades DataFrame",
            False,
            f"Exception with empty trades: {e}"
        ))

    return results


def verify_monthly_heatmap() -> List[TestResult]:
    """Verify create_monthly_heatmap() implementation."""
    results = []
    result = create_sample_result()

    # Test 1: Basic functionality
    try:
        fig = ChartFactory.create_monthly_heatmap(result)
        results.append(TestResult(
            "create_monthly_heatmap returns go.Figure",
            isinstance(fig, go.Figure),
            f"Expected go.Figure, got {type(fig)}"
        ))
    except Exception as e:
        results.append(TestResult(
            "create_monthly_heatmap returns go.Figure",
            False,
            f"Exception: {e}"
        ))
        return results

    # Test 2: Has heatmap trace
    has_heatmap = any(isinstance(trace, go.Heatmap) for trace in fig.data)
    results.append(TestResult(
        "Monthly heatmap has Heatmap trace",
        has_heatmap,
        "Heatmap trace found" if has_heatmap else "No heatmap trace found"
    ))

    # Test 3: Has title
    has_title = fig.layout.title is not None and 'Monthly' in str(fig.layout.title.text)
    results.append(TestResult(
        "Monthly heatmap has title",
        has_title,
        f"Title: {fig.layout.title.text if fig.layout.title else 'None'}"
    ))

    # Test 4: Has axis labels
    has_xaxis = fig.layout.xaxis.title is not None
    has_yaxis = fig.layout.yaxis.title is not None
    results.append(TestResult(
        "Monthly heatmap has axis labels",
        has_xaxis and has_yaxis,
        f"X-axis: {fig.layout.xaxis.title.text if has_xaxis else 'None'}, Y-axis: {fig.layout.yaxis.title.text if has_yaxis else 'None'}"
    ))

    # Test 5: Uses RdYlGn colorscale
    heatmap_trace = next((trace for trace in fig.data if isinstance(trace, go.Heatmap)), None)
    if heatmap_trace:
        uses_rdylgn = 'RdYlGn' in str(heatmap_trace.colorscale)
        results.append(TestResult(
            "Monthly heatmap uses RdYlGn colorscale",
            uses_rdylgn,
            f"Colorscale: {heatmap_trace.colorscale}"
        ))

        # Test 6: Centered at 0
        zmid_zero = heatmap_trace.zmid == 0
        results.append(TestResult(
            "Monthly heatmap centered at 0 (zmid=0)",
            zmid_zero,
            f"zmid: {heatmap_trace.zmid}"
        ))

    # Test 7: Handles short data gracefully
    try:
        short_result = create_sample_result()
        short_result.equity_curve = np.array([100000, 100500])
        short_result.drawdown_curve = np.array([0.0, 0.0])
        fig_short = ChartFactory.create_monthly_heatmap(short_result)
        results.append(TestResult(
            "Monthly heatmap handles short data (< 1 month)",
            isinstance(fig_short, go.Figure),
            "Returns figure with insufficient data message"
        ))
    except Exception as e:
        results.append(TestResult(
            "Monthly heatmap handles short data (< 1 month)",
            False,
            f"Exception with short data: {e}"
        ))

    return results


def verify_consistent_styling() -> List[TestResult]:
    """Verify consistent styling across all charts."""
    results = []
    result = create_sample_result()
    close_prices = np.linspace(100, 110, len(result.equity_curve))

    # Create all charts
    equity_fig = ChartFactory.create_equity_curve(result)
    dd_fig = ChartFactory.create_drawdown_chart(result)
    trades_fig = ChartFactory.create_trades_chart(result, close_prices)
    heatmap_fig = ChartFactory.create_monthly_heatmap(result)

    # Test 1: All charts have titles
    all_have_titles = all([
        equity_fig.layout.title is not None,
        dd_fig.layout.title is not None,
        trades_fig.layout.title is not None,
        heatmap_fig.layout.title is not None,
    ])
    results.append(TestResult(
        "All charts have titles",
        all_have_titles,
        "All charts have titles" if all_have_titles else "Some charts missing titles"
    ))

    # Test 2: All charts have axis labels
    all_have_axes = all([
        equity_fig.layout.xaxis.title is not None and equity_fig.layout.yaxis.title is not None,
        dd_fig.layout.xaxis.title is not None and dd_fig.layout.yaxis.title is not None,
        trades_fig.layout.xaxis.title is not None and trades_fig.layout.yaxis.title is not None,
        heatmap_fig.layout.xaxis.title is not None and heatmap_fig.layout.yaxis.title is not None,
    ])
    results.append(TestResult(
        "All charts have axis labels",
        all_have_axes,
        "All charts have axis labels" if all_have_axes else "Some charts missing axis labels"
    ))

    # Test 3: Dark theme applies to all
    equity_dark = ChartFactory.create_equity_curve(result, dark_theme=True)
    dd_dark = ChartFactory.create_drawdown_chart(result, dark_theme=True)
    trades_dark = ChartFactory.create_trades_chart(result, close_prices, dark_theme=True)
    heatmap_dark = ChartFactory.create_monthly_heatmap(result, dark_theme=True)

    all_dark = all([
        'dark' in str(equity_dark.layout.template).lower(),
        'dark' in str(dd_dark.layout.template).lower(),
        'dark' in str(trades_dark.layout.template).lower(),
        'dark' in str(heatmap_dark.layout.template).lower(),
    ])
    results.append(TestResult(
        "All charts support dark_theme parameter",
        all_dark,
        "All charts apply dark theme" if all_dark else "Some charts don't apply dark theme"
    ))

    return results


def verify_exports() -> List[TestResult]:
    """Verify ChartFactory is properly exported."""
    results = []

    # Test 1: ChartFactory in __all__
    try:
        from simple_futures_backtester.output import ChartFactory as ImportedFactory
        results.append(TestResult(
            "ChartFactory exported from output package",
            True,
            "Successfully imported from simple_futures_backtester.output"
        ))
    except ImportError as e:
        results.append(TestResult(
            "ChartFactory exported from output package",
            False,
            f"Import failed: {e}"
        ))

    # Test 2: ChartFactory direct import
    try:
        from simple_futures_backtester.output.charts import ChartFactory as DirectFactory
        results.append(TestResult(
            "ChartFactory importable from charts module",
            True,
            "Successfully imported from simple_futures_backtester.output.charts"
        ))
    except ImportError as e:
        results.append(TestResult(
            "ChartFactory importable from charts module",
            False,
            f"Import failed: {e}"
        ))

    return results


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("I4.T4 VERIFICATION: Chart Generation Module")
    print("=" * 80)
    print()

    all_results: List[TestResult] = []

    sections = [
        ("Equity Curve Chart", verify_equity_curve),
        ("Drawdown Chart", verify_drawdown_chart),
        ("Trades Chart", verify_trades_chart),
        ("Monthly Heatmap", verify_monthly_heatmap),
        ("Consistent Styling", verify_consistent_styling),
        ("Package Exports", verify_exports),
    ]

    for section_name, verify_func in sections:
        print(f"\n{section_name}")
        print("-" * 80)
        section_results = verify_func()
        all_results.extend(section_results)

        for test in section_results:
            status = "✓ PASS" if test.passed else "✗ FAIL"
            print(f"{status}: {test.name}")
            if not test.passed:
                print(f"         {test.message}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if failed == 0:
        print("\n✓ All acceptance criteria met!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        print("\nFailed tests:")
        for test in all_results:
            if not test.passed:
                print(f"  - {test.name}: {test.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
