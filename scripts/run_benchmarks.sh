#!/bin/bash
# Benchmark Execution Script
#
# Runs the pytest-benchmark suite for performance validation.
#
# Usage: ./scripts/run_benchmarks.sh [suite]
#   suite: all, bars, indicators, backtest (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

SUITE="${1:-all}"

echo "Running benchmarks: $SUITE"
echo "================================"

case "$SUITE" in
    all)
        pytest tests/benchmarks/ -m benchmark --benchmark-only -v
        ;;
    bars)
        pytest tests/benchmarks/bench_bars.py -m benchmark --benchmark-only -v
        ;;
    indicators)
        pytest tests/benchmarks/bench_indicators.py -m benchmark --benchmark-only -v
        ;;
    backtest)
        pytest tests/benchmarks/bench_backtest.py -m benchmark --benchmark-only -v
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Usage: $0 [all|bars|indicators|backtest]"
        exit 1
        ;;
esac

echo "================================"
echo "Benchmarks complete!"
