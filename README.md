# Simple Futures Backtester

A high-performance, vectorized futures backtesting framework built on a forked VectorBT repository. Maximizes reuse of battle-tested indicators and portfolio analytics while adding custom extensions for delayed trailing stops and alternative bar types.

## Features

- **High Performance**: JIT-compiled bar factories achieving 1M+ rows/sec generation
- **Alternative Bar Types**: Renko, Range, Tick, Volume, Dollar, and Imbalance bars
- **VectorBT Integration**: Full access to RSI, MA, MACD, ATR, Bollinger Bands, and more
- **Futures-Specific**: Point value application at analytics extraction
- **Trailing Stops**: Delayed and ATR-based trailing stop implementations
- **Parameter Sweeps**: Grid search optimization via ProcessPoolExecutor
- **Rich CLI**: Typer-based commands with Rich console output
- **Export Options**: PNG charts, CSV data, and JSON reports via Plotly

## Requirements

- Python 3.11 or higher
- Dependencies are automatically installed via pip

## Installation

### Step 1: Clone and Install Package

```bash
git clone https://github.com/simple-futures-backtester/simple-futures-backtester.git
cd simple_futures_backtester

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev]"
```

### Step 2: Install VectorBT Fork (Required)

The project depends on VectorBT v0.26.2 installed as an editable fork:

```bash
# Run the installation script
./scripts/install_fork.sh
```

**What this script does:**
1. Downloads VectorBT v0.26.2 source from PyPI
2. Extracts it to `lib/vectorbt/`
3. Initializes a git repository for local patch tracking
4. Installs in editable mode (`pip install -e lib/vectorbt/`)

**Manual Installation (if script fails):**

```bash
# Create lib directory
mkdir -p lib

# Download and extract VectorBT v0.26.2
pip download vectorbt==0.26.2 --no-deps --no-binary :all: -d /tmp
tar -xzf /tmp/vectorbt-0.26.2.tar.gz -C lib
mv lib/vectorbt-0.26.2 lib/vectorbt

# Install in editable mode
pip install -e lib/vectorbt/
```

**Verify Installation:**

```bash
python -c "import vectorbt as vbt; print(f'VectorBT version: {vbt.__version__}')"
# Expected: VectorBT version: 0.26.2
```

### Troubleshooting VectorBT Installation

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'vectorbt'` | Run `pip install -e lib/vectorbt/` again |
| Version mismatch (not 0.26.2) | Clear `PYTHONPATH` and reinstall fork |
| Import conflicts | Remove any system-level vectorbt: `pip uninstall vectorbt` |

## Quick Start

### Running Your First Backtest

```bash
# Using the included sample data
sfb backtest \
    --strategy momentum \
    --data examples/sample_data/es_1min_sample.csv

# With a YAML configuration file
sfb backtest \
    --strategy momentum \
    --config configs/strategies/momentum.yaml \
    --data examples/sample_data/es_1min_sample.csv
```

**Expected Output:**

```
Loading data...
Loaded 1000 bars
Initializing strategy: momentum
Generating signals...
Generated 1000 signals
Running backtest...
Backtest complete

                  Backtest Results
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃           Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Total Return       │         +2.45%  │
│ Sharpe Ratio       │         +0.8234 │
│ Sortino Ratio      │         +1.1456 │
│ Max Drawdown       │         -5.32%  │
│ Win Rate           │          58.00% │
│ Profit Factor      │          1.4500 │
│ Number of Trades   │              25 │
│ Average Trade      │         +0.0098 │
└────────────────────┴─────────────────┘
```

### Running a Parameter Sweep

```bash
# Optimize RSI and EMA parameters
sfb sweep \
    --strategy momentum \
    --sweep-config configs/sweeps/momentum_sweep.yaml \
    --data examples/sample_data/es_1min_sample.csv \
    --n-jobs -1 \
    --output output/sweep_results
```

### Generating Alternative Bars

```bash
# Generate Renko bars with 10-point brick size
sfb generate-bars \
    --bar-type renko \
    --param 10 \
    --data examples/sample_data/es_1min_sample.csv \
    --output renko_bars.csv

# Generate Volume bars with 5000-share threshold
sfb generate-bars \
    --bar-type volume \
    --param 5000 \
    --data examples/sample_data/es_1min_sample.csv
```

### Running Benchmarks

```bash
# Run all benchmarks
sfb benchmark --suite full

# Run bar generation benchmarks only
sfb benchmark --suite bars
```

## CLI Command Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `sfb backtest` | Run a single backtest | `--strategy`, `--data`, `--config`, `--capital`, `--output` |
| `sfb sweep` | Parameter grid search | `--strategy`, `--sweep-config`, `--data`, `--n-jobs`, `--output` |
| `sfb generate-bars` | Generate alternative bars | `--bar-type`, `--param`, `--data`, `--output` |
| `sfb benchmark` | Performance benchmarks | `--suite` (full, bars, backtest, indicators) |
| `sfb export` | Export results to files | `--input`, `--output`, `--format` (all, png, html, csv) |
| `sfb version` | Show version info | - |

### Detailed Command Options

**`sfb backtest`**
```
Options:
  -s, --strategy TEXT   Strategy name (momentum, mean_reversion, breakout) [required]
  -d, --data TEXT       Path to OHLCV data file (CSV/Parquet) [required]
  -c, --config TEXT     Path to strategy config YAML
  --capital FLOAT       Override initial capital (default: 100000)
  --fees FLOAT          Override transaction fees (default: 0.0001)
  --slippage FLOAT      Override slippage (default: 0.0001)
  -o, --output TEXT     Output directory for results
```

**`sfb sweep`**
```
Options:
  -s, --strategy TEXT      Strategy name [required]
  -c, --sweep-config TEXT  Path to sweep configuration YAML [required]
  -d, --data TEXT          Path to OHLCV data file [required]
  -j, --n-jobs INTEGER     Parallel workers (-1 = all cores, default: 1)
  -o, --output TEXT        Output directory for results
```

**`sfb generate-bars`**
```
Options:
  -t, --bar-type TEXT  Bar type: renko, range, tick, volume, dollar,
                       tick_imbalance, volume_imbalance [required]
  -p, --param TEXT     Bar parameter value (e.g., brick_size for renko) [required]
  -d, --data TEXT      Path to OHLCV data file [required]
  -o, --output TEXT    Output CSV file path
```

Run `sfb --help` or `sfb <command> --help` for complete documentation.

## Project Structure

```
simple_futures_backtester/
├── simple_futures_backtester/   # Main package
│   ├── cli.py                   # Typer CLI entry point
│   ├── config.py                # Configuration loading
│   ├── data/                    # Data loading and validation
│   ├── bars/                    # Alternative bar generators
│   ├── extensions/              # Trailing stops, futures portfolio
│   ├── strategy/                # Strategy framework and examples
│   ├── backtest/                # Backtest engine and sweeps
│   ├── output/                  # Reports and exports
│   └── utils/                   # JIT utilities, logging
├── lib/vectorbt/                # Forked VectorBT v0.26.2
├── configs/                     # Strategy and sweep configurations
├── tests/                       # Test suite with benchmarks
├── docs/                        # Documentation and diagrams
└── scripts/                     # Installation and utility scripts
```

## Configuration

Strategy configurations are YAML files:

```yaml
strategy:
  name: momentum
  class: simple_futures_backtester.strategy.examples.momentum.MomentumStrategy

parameters:
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  ema_fast: 12
  ema_slow: 26

backtest:
  initial_cash: 100000
  point_value: 50.0
  commission: 2.5
  slippage: 0.5
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simple_futures_backtester --cov-report=html

# Run benchmarks
pytest -m benchmark --benchmark-only
```

### Code Quality

```bash
# Format code
black simple_futures_backtester tests

# Lint code
ruff check simple_futures_backtester tests

# Type checking
mypy simple_futures_backtester
```

## Performance Targets

| Operation | Target |
|-----------|--------|
| Bar generation | 1M+ rows/sec |
| Single backtest | <50ms |
| 100-combo sweep | <10s |

## Documentation

- [Quick Start Guide](docs/guides/quickstart.md) - Step-by-step tutorial with sample data
- [Strategy Development](docs/guides/strategies.md) - How to implement custom strategies
- [Alternative Bar Types](docs/guides/bar_types.md) - All 7 bar types with use cases
- [API Reference](docs/api/reference.md) - Complete Python API documentation

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
