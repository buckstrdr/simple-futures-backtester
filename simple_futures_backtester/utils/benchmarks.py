"""Benchmark utilities for programmatic pytest-benchmark execution.

Provides functions to run benchmark suites, parse results, and compare
against performance targets. Used by the CLI benchmark command.

Performance Targets (from plan_manifest.json):
- Bar generation: 1M+ rows/sec
- Single backtest: <50ms
- Parameter sweep (100 combos): <10 seconds
- Data loading (1M rows): <2 seconds
- Peak memory (5M rows): <2GB

Usage:
    >>> from simple_futures_backtester.utils.benchmarks import (
    ...     run_benchmark_suite,
    ...     parse_benchmark_output,
    ...     check_benchmark_status,
    ...     create_benchmark_table,
    ... )
    >>>
    >>> # Run benchmarks
    >>> exit_code, stdout = run_benchmark_suite("full")
    >>> results = parse_benchmark_output(stdout)
    >>> table = create_benchmark_table(results)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    pass


# Suite name to pytest marker expression mapping
SUITE_MAP: dict[str, str] = {
    "full": "benchmark",
    "bars": "benchmark and bars",
    "backtest": "benchmark and backtest",
    "indicators": "benchmark and indicators",
}

# Performance targets from plan_manifest.json
# Key: component name, Value: (target_value, unit, comparison_direction)
# comparison_direction: "higher" = higher is better (throughput)
#                       "lower" = lower is better (latency)
PERFORMANCE_TARGETS: dict[str, tuple[float, str, str]] = {
    "bar_generation": (1_000_000.0, "rows/sec", "higher"),
    "single_backtest": (50.0, "ms", "lower"),
    "parameter_sweep_100": (10.0, "seconds", "lower"),
    "data_loading_1M": (2.0, "seconds", "lower"),
    "peak_memory_5M": (2048.0, "MB", "lower"),
    # Additional bar-specific targets
    "renko_bars": (1_000_000.0, "rows/sec", "higher"),
    "range_bars": (1_000_000.0, "rows/sec", "higher"),
    "tick_bars": (1_000_000.0, "rows/sec", "higher"),
    "volume_bars": (1_000_000.0, "rows/sec", "higher"),
    "dollar_bars": (1_000_000.0, "rows/sec", "higher"),
    "imbalance_bars": (1_000_000.0, "rows/sec", "higher"),
}


def get_available_suites() -> list[str]:
    """Get list of available benchmark suite names.

    Returns:
        List of suite names that can be passed to run_benchmark_suite().
    """
    return list(SUITE_MAP.keys())


def run_benchmark_suite(
    suite: str,
    tests_dir: str | Path | None = None,
) -> tuple[int, str, str]:
    """Run pytest-benchmark for the specified suite.

    Executes pytest with benchmark-specific options and captures output.
    Uses JSON output for reliable parsing of results.

    Args:
        suite: Suite name from SUITE_MAP (full, bars, backtest, indicators).
        tests_dir: Optional path to tests directory. Defaults to
            discovering tests/benchmarks/ relative to package.

    Returns:
        tuple[int, str, str]: (exit_code, stdout, stderr)
            exit_code: 0 if all tests passed, non-zero otherwise
            stdout: Standard output from pytest
            stderr: Standard error from pytest
    """
    # Get marker expression for suite
    marker_expr = SUITE_MAP.get(suite)
    if marker_expr is None:
        available = ", ".join(SUITE_MAP.keys())
        return (1, "", f"Unknown suite '{suite}'. Available: {available}")

    # Determine tests directory
    if tests_dir is None:
        # Try to find tests/benchmarks relative to package or cwd
        possible_paths = [
            Path.cwd() / "tests" / "benchmarks",
            Path(__file__).parent.parent.parent / "tests" / "benchmarks",
        ]
        for path in possible_paths:
            if path.exists():
                tests_dir = path
                break
        else:
            return (0, "No benchmark tests found in tests/benchmarks/", "")

    tests_dir = Path(tests_dir)
    if not tests_dir.exists():
        return (0, f"Benchmark directory not found: {tests_dir}", "")

    # Check if any benchmark files exist
    bench_files = list(tests_dir.glob("bench_*.py"))
    if not bench_files:
        return (0, "No benchmark test files found (bench_*.py)", "")

    # Create temp file for JSON output
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as tmp:
        json_output_path = tmp.name

    try:
        # Check if pytest-benchmark is available
        check_benchmark = subprocess.run(
            [sys.executable, "-c", "import pytest_benchmark"],
            capture_output=True,
            text=True,
        )
        has_benchmark = check_benchmark.returncode == 0

        # Build pytest command
        # Note: bench_*.py files may not have @pytest.mark.benchmark markers
        # So we run tests from the benchmarks directory directly and parse output
        # Override pytest.ini_options to discover bench_*.py files
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(tests_dir),
            "-v",
            "--tb=short",
            "-s",  # Capture print output for throughput parsing
            "-o",
            "python_files=bench_*.py test_*.py",  # Override to include bench_* files
        ]

        # Add marker filter if not running full suite
        # Only add marker if suite has specific markers defined in tests
        if suite != "full" and has_benchmark:
            cmd.extend(["-m", marker_expr])

        # Add benchmark-specific flags if pytest-benchmark is available
        if has_benchmark:
            cmd.extend(
                [
                    "--benchmark-disable-gc",
                    "--benchmark-warmup=on",
                    f"--benchmark-json={json_output_path}",
                ]
            )

        # Run pytest
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            timeout=600,  # 10 minute timeout
        )

        # Read JSON output if it exists
        json_output = ""
        json_path = Path(json_output_path)
        if json_path.exists():
            json_output = json_path.read_text()
            json_path.unlink()

        # Combine stdout with JSON output marker
        stdout = result.stdout
        if json_output:
            stdout += f"\n---BENCHMARK_JSON_START---\n{json_output}\n---BENCHMARK_JSON_END---"

        return (result.returncode, stdout, result.stderr)

    except subprocess.TimeoutExpired:
        return (1, "", "Benchmark timed out after 10 minutes")
    except FileNotFoundError:
        return (1, "", "pytest not found. Install with: pip install pytest pytest-benchmark")
    finally:
        # Cleanup temp file if still exists
        json_path = Path(json_output_path)
        if json_path.exists():
            json_path.unlink()


def parse_benchmark_output(stdout: str) -> list[dict[str, str | float]]:
    """Parse pytest-benchmark output to extract metrics.

    Attempts to parse JSON output if available, falls back to text parsing.

    Args:
        stdout: Standard output from pytest-benchmark execution.

    Returns:
        list[dict]: List of benchmark results with keys:
            - component: Component name (e.g., "range_bars")
            - metric: Metric name (e.g., "throughput")
            - actual: Actual value (float)
            - unit: Unit (e.g., "rows/sec")
            - test_name: Original test name
    """
    results: list[dict[str, str | float]] = []

    # Try to extract JSON output
    if "---BENCHMARK_JSON_START---" in stdout:
        try:
            json_start = stdout.index("---BENCHMARK_JSON_START---") + len(
                "---BENCHMARK_JSON_START---"
            )
            json_end = stdout.index("---BENCHMARK_JSON_END---")
            json_str = stdout[json_start:json_end].strip()
            bench_data = json.loads(json_str)

            # Parse benchmarks from JSON
            for bench in bench_data.get("benchmarks", []):
                name = bench.get("name", "unknown")
                stats = bench.get("stats", {})

                # Extract component name from test name
                # e.g., "test_range_bars_1m_rows_performance" -> "range_bars"
                component = _extract_component_name(name)

                # Calculate throughput if possible
                mean_time = stats.get("mean", 0)
                if mean_time > 0:
                    # Check if this is a throughput test (1M rows)
                    if "1m_rows" in name.lower() or "1m" in name.lower():
                        throughput = 1_000_000 / mean_time
                        results.append(
                            {
                                "component": component,
                                "metric": "throughput",
                                "actual": throughput,
                                "unit": "rows/sec",
                                "test_name": name,
                            }
                        )
                    else:
                        # Report as latency in ms
                        latency_ms = mean_time * 1000
                        results.append(
                            {
                                "component": component,
                                "metric": "latency",
                                "actual": latency_ms,
                                "unit": "ms",
                                "test_name": name,
                            }
                        )

            return results

        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Fallback: Parse text output for throughput patterns
    # Look for patterns like "Range bars throughput: 1,234,567 rows/sec"
    import re

    throughput_pattern = re.compile(
        r"(\w+(?:\s+\w+)*)\s+throughput:\s+([\d,]+)\s+rows/sec",
        re.IGNORECASE,
    )

    for match in throughput_pattern.finditer(stdout):
        component_raw = match.group(1).lower().replace(" ", "_")
        throughput_str = match.group(2).replace(",", "")
        try:
            throughput = float(throughput_str)
            results.append(
                {
                    "component": component_raw,
                    "metric": "throughput",
                    "actual": throughput,
                    "unit": "rows/sec",
                    "test_name": f"test_{component_raw}",
                }
            )
        except ValueError:
            pass

    return results


def _extract_component_name(test_name: str) -> str:
    """Extract component name from test function name.

    Args:
        test_name: Full test name (e.g., "test_range_bars_1m_rows_performance")

    Returns:
        Component name (e.g., "range_bars")
    """
    # Remove "test_" prefix
    name = test_name.lower()
    if name.startswith("test_"):
        name = name[5:]

    # Common suffixes to remove
    suffixes = [
        "_1m_rows_performance",
        "_100k_rows",
        "_performance",
        "_benchmark",
        "_throughput",
        "_latency",
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def check_benchmark_status(
    component: str,
    actual: float,
    metric: str = "throughput",  # noqa: ARG001
) -> tuple[str, str, float | None]:
    """Check if benchmark passes target threshold.

    Args:
        component: Component name from PERFORMANCE_TARGETS.
        actual: Actual measured value.
        metric: Metric type ("throughput" or "latency").

    Returns:
        tuple[str, str, float | None]: (status_text, status_color, target_value)
            status_text: "PASS", "FAIL", or "N/A"
            status_color: "green", "red", or "yellow"
            target_value: Target threshold or None if not found
    """
    # Look up target by component name
    target_info = PERFORMANCE_TARGETS.get(component)

    # Try variations of the component name
    if target_info is None:
        # Try without "_bars" suffix
        alt_name = component.replace("_bars", "")
        target_info = PERFORMANCE_TARGETS.get(f"{alt_name}_bars")

    # Try generic "bar_generation" for any bar type
    if target_info is None and "bar" in component.lower():
        target_info = PERFORMANCE_TARGETS.get("bar_generation")

    if target_info is None:
        return ("N/A", "yellow", None)

    target_value, _unit, direction = target_info

    # Compare based on direction
    passed = actual >= target_value if direction == "higher" else actual <= target_value

    if passed:
        return ("PASS", "green", target_value)
    else:
        return ("FAIL", "red", target_value)


def create_benchmark_table(
    results: list[dict[str, str | float]],
    title: str = "Benchmark Results",
) -> Table:
    """Create Rich table with benchmark results.

    Args:
        results: List of benchmark result dicts from parse_benchmark_output().
        title: Table title.

    Returns:
        Rich Table object ready for console.print().
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan", no_wrap=True, width=20)
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Actual", justify="right", width=18)
    table.add_column("Target", justify="right", width=18)
    table.add_column("Status", justify="center", width=8)

    for result in results:
        component = str(result.get("component", "unknown"))
        metric = str(result.get("metric", "unknown"))
        actual = float(result.get("actual", 0))
        unit = str(result.get("unit", ""))

        # Get status and target
        status_text, status_color, target_value = check_benchmark_status(component, actual, metric)

        # Format actual value
        if actual >= 1_000_000:
            actual_str = f"{actual / 1_000_000:.2f}M {unit}"
        elif actual >= 1_000:
            actual_str = f"{actual / 1_000:.2f}K {unit}"
        else:
            actual_str = f"{actual:.2f} {unit}"

        # Format target value
        if target_value is not None:
            if target_value >= 1_000_000:
                target_str = f"{target_value / 1_000_000:.0f}M+ {unit}"
            elif target_value >= 1_000:
                target_str = f"<{target_value / 1_000:.0f}K {unit}"
            else:
                target_str = f"<{target_value:.0f} {unit}"
        else:
            target_str = "N/A"

        table.add_row(
            component,
            metric,
            actual_str,
            target_str,
            f"[{status_color}]{status_text}[/{status_color}]",
        )

    return table


def check_all_passed(results: list[dict[str, str | float]]) -> bool:
    """Check if all benchmark results pass their targets.

    Args:
        results: List of benchmark result dicts from parse_benchmark_output().

    Returns:
        True if all benchmarks pass or have no target, False if any fail.
    """
    for result in results:
        component = str(result.get("component", "unknown"))
        actual = float(result.get("actual", 0))
        metric = str(result.get("metric", "unknown"))

        status_text, _, _ = check_benchmark_status(component, actual, metric)
        if status_text == "FAIL":
            return False

    return True


__all__ = [
    "PERFORMANCE_TARGETS",
    "SUITE_MAP",
    "check_all_passed",
    "check_benchmark_status",
    "create_benchmark_table",
    "get_available_suites",
    "parse_benchmark_output",
    "run_benchmark_suite",
]
