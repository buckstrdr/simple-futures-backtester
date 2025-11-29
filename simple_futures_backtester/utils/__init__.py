"""Utility modules for JIT compilation, logging, and benchmarks.

Provides Numba decorators, dtype helpers, structured logging utilities,
and benchmark execution utilities.
"""

from simple_futures_backtester.utils.benchmarks import (
    PERFORMANCE_TARGETS,
    SUITE_MAP,
    check_all_passed,
    check_benchmark_status,
    create_benchmark_table,
    get_available_suites,
    parse_benchmark_output,
    run_benchmark_suite,
)
from simple_futures_backtester.utils.jit_utils import (
    ensure_float64,
    ensure_int64,
    get_njit_decorator,
    is_contiguous,
    preallocate_2d_array,
    preallocate_array,
    validate_ohlcv_arrays,
)

__all__: list[str] = [
    # JIT utilities
    "ensure_float64",
    "ensure_int64",
    "get_njit_decorator",
    "is_contiguous",
    "preallocate_2d_array",
    "preallocate_array",
    "validate_ohlcv_arrays",
    # Benchmark utilities
    "PERFORMANCE_TARGETS",
    "SUITE_MAP",
    "check_all_passed",
    "check_benchmark_status",
    "create_benchmark_table",
    "get_available_suites",
    "parse_benchmark_output",
    "run_benchmark_suite",
]
