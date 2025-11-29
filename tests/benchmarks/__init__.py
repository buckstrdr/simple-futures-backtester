"""Performance benchmarks for Simple Futures Backtester.

Contains pytest-benchmark tests for:
- Bar generation throughput (target: 1M+ rows/sec)
- Backtest engine execution (target: <50ms single backtest)
- Parameter sweep performance (target: <10 seconds for 100 combos)

Benchmark Files:
    bench_bars.py: Consolidated bar generation benchmarks (Renko, Range, Tick,
        Volume, Dollar, Tick Imbalance, Volume Imbalance)
    bench_backtest.py: Single backtest latency benchmarks
    bench_sweep.py: Parameter sweep performance benchmarks

    Individual bar benchmarks (legacy):
    bench_range_bars.py, bench_tick_bars.py, bench_volume_bars.py,
    bench_dollar_bars.py, bench_imbalance_bars.py

Baselines:
    baselines/targets.json: Performance targets for CI comparison

Usage:
    # Run all benchmarks
    pytest tests/benchmarks/ -v

    # Run only bar benchmarks
    pytest tests/benchmarks/ -m "benchmark and bars"

    # Run only backtest benchmarks
    pytest tests/benchmarks/ -m "benchmark and backtest"

    # Run benchmarks with detailed output
    pytest tests/benchmarks/ -v -s

Pytest Markers:
    @pytest.mark.benchmark: All benchmark tests
    @pytest.mark.bars: Bar generation benchmarks
    @pytest.mark.backtest: Backtest and sweep benchmarks
"""
