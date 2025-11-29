# Quick Start Guide

This guide walks you through installing Simple Futures Backtester and running your first backtest in under 5 minutes.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- git (for cloning the repository)

## Installation

### Step 1: Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/simple-futures-backtester/simple-futures-backtester.git
cd simple_futures_backtester

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install package with development dependencies
pip install -e ".[dev]"
```

### Step 2: Install VectorBT Fork

The backtester uses VectorBT v0.26.2 for technical indicators. Install it using the provided script:

```bash
./scripts/install_fork.sh
```

**Expected Output:**

```
Installing VectorBT v0.26.2 fork...
Using project virtual environment: /path/to/.venv
Downloading VectorBT v0.26.2 from PyPI...
Extracting source...
Initializing git repository for fork tracking...
Installing VectorBT in editable mode...

VectorBT v0.26.2 fork installation complete!
Location: /path/to/lib/vectorbt

Verification:
  VectorBT version: 0.26.2
```

### Step 3: Verify Installation

```bash
# Check CLI is available
sfb --help

# Verify VectorBT import
python -c "import vectorbt as vbt; print(f'VectorBT: {vbt.__version__}')"
```

**Expected Output:**

```
Usage: sfb [OPTIONS] COMMAND [ARGS]...

  Simple Futures Backtester - High-performance vectorized backtesting framework.

Commands:
  backtest       Run a single backtest with a strategy configuration.
  benchmark      Run performance benchmarks and display results vs targets.
  export         Export backtest results to various formats.
  generate-bars  Generate alternative bar types from OHLCV data.
  sweep          Execute parameter grid search optimization.
  version        Show version information.

VectorBT: 0.26.2
```

## Running Your First Backtest

### Using Sample Data

The project includes sample E-mini S&P 500 futures data for testing:

```bash
sfb backtest \
    --strategy momentum \
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
├────────────────────┼─────────────────┤
│ Config Hash        │ 7a3b2c1d...     │
│ Timestamp          │ 2024-01-15T...  │
└────────────────────┴─────────────────┘
```

### Understanding the Results

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Total Return** | Net profit/loss as percentage | Positive is good |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 is good, > 2.0 is excellent |
| **Sortino Ratio** | Downside risk-adjusted return | > 1.5 is good |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% is acceptable |
| **Win Rate** | Percentage of profitable trades | > 50% is typical |
| **Profit Factor** | Gross profits / Gross losses | > 1.5 is good |
| **Number of Trades** | Total closed trades | Depends on strategy |
| **Average Trade** | Mean profit per trade | Positive is good |

### Using a Configuration File

For more control, use a YAML configuration file:

```bash
sfb backtest \
    --strategy momentum \
    --config configs/strategies/momentum.yaml \
    --data examples/sample_data/es_1min_sample.csv
```

**Example Configuration (`configs/strategies/momentum.yaml`):**

```yaml
strategy:
  name: momentum
  parameters:
    rsi_period: 14
    fast_ema: 9
    slow_ema: 21

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

### Exporting Results

Export backtest results to files for further analysis:

```bash
sfb backtest \
    --strategy momentum \
    --data examples/sample_data/es_1min_sample.csv \
    --output output/my_backtest
```

This creates:

```
output/my_backtest/
├── charts/
│   ├── equity.png      # Equity curve chart
│   ├── equity.html     # Interactive equity chart
│   ├── drawdown.png    # Drawdown chart
│   ├── drawdown.html   # Interactive drawdown chart
│   ├── monthly.png     # Monthly returns heatmap
│   └── monthly.html    # Interactive heatmap
├── data/
│   ├── equity_curve.csv    # Time series data
│   ├── trades.csv          # All trade details
│   ├── metrics.csv         # Summary metrics
│   └── monthly_returns.csv # Monthly returns
└── report.json             # JSON summary
```

## Running a Parameter Sweep

Optimize strategy parameters by testing all combinations:

### Step 1: Create Sweep Configuration

Create `configs/sweeps/momentum_sweep.yaml`:

```yaml
sweep:
  strategy: momentum
  parameters:
    rsi_period: [10, 14, 20]
    fast_ema: [5, 9, 12]
    slow_ema: [21, 30, 50]

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

### Step 2: Run the Sweep

```bash
sfb sweep \
    --strategy momentum \
    --sweep-config configs/sweeps/momentum_sweep.yaml \
    --data examples/sample_data/es_1min_sample.csv \
    --n-jobs -1 \
    --output output/sweep_results
```

**Expected Output:**

```
Loading data...
Loaded 1000 bars
Loading sweep configuration...
Loaded config: configs/sweeps/momentum_sweep.yaml
Validating strategy: momentum
Testing 27 parameter combinations with 8 worker(s)...
Testing combinations ━━━━━━━━━━━━━━━━━━━━━ 100% (27/27) 0:00:05
Sweep complete

   Top 10 Parameter Combinations by Sharpe Ratio
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Rank ┃    Sharpe ┃   Return ┃   Max DD ┃ Trades ┃   rsi_period ┃ fast_ema ┃ slow_ema ┃
┡━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│  1   │   1.2345 │   +5.67% │   -8.12% │     32 │           14 │        9 │       21 │
│  2   │   1.1234 │   +4.89% │   -7.45% │     28 │           10 │        9 │       30 │
│  3   │   1.0567 │   +4.23% │   -6.78% │     25 │           20 │       12 │       50 │
│ ...  │      ... │      ... │      ... │    ... │          ... │      ... │      ... │
└──────┴──────────┴──────────┴──────────┴────────┴──────────────┴──────────┴──────────┘

Best parameters: rsi_period=14, fast_ema=9, slow_ema=21
Best Sharpe ratio: 1.2345
```

## Generating Alternative Bars

Transform time-based OHLCV data into alternative bar types:

```bash
# Generate Renko bars (price-based)
sfb generate-bars \
    --bar-type renko \
    --param 5.0 \
    --data examples/sample_data/es_1min_sample.csv \
    --output output/renko_bars.csv

# Generate Volume bars (activity-based)
sfb generate-bars \
    --bar-type volume \
    --param 10000 \
    --data examples/sample_data/es_1min_sample.csv \
    --output output/volume_bars.csv
```

**Expected Output:**

```
Loading data...
Loaded 1000 bars
Generating renko bars with brick_size=5.0...
Generated 156 bars from 1000 source bars
Compression ratio: 6.41:1 (1000 -> 156 bars)
Saved 156 bars to output/renko_bars.csv
```

## Using Your Own Data

### Data Format Requirements

Your OHLCV data must be a CSV or Parquet file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime | Bar timestamp (parsed with pandas) |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | int | Trading volume |

**Example CSV:**

```csv
datetime,open,high,low,close,volume
2024-01-02 09:30:00,4750.25,4752.50,4749.00,4751.75,12500
2024-01-02 09:31:00,4751.75,4753.00,4750.50,4752.25,8750
2024-01-02 09:32:00,4752.25,4754.00,4751.00,4753.50,15000
```

### Running with Your Data

```bash
sfb backtest \
    --strategy momentum \
    --data /path/to/your/data.csv
```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'vectorbt'`**

```bash
# Reinstall the VectorBT fork
pip install -e lib/vectorbt/
```

**Issue: `Error: Unknown strategy 'my_strategy'`**

The strategy must be registered. Available built-in strategies:
- `momentum` - RSI + EMA crossover
- `mean_reversion` - Bollinger Bands
- `breakout` - EMA crossover

**Issue: `Data validation error: Missing required columns`**

Ensure your CSV has all required columns: `datetime`, `open`, `high`, `low`, `close`, `volume`

**Issue: `No signals generated` or empty results**

- Check that your data covers enough bars for indicator warmup
- RSI needs at least `rsi_period` bars before generating signals
- Try adjusting strategy parameters

### Getting Help

```bash
# Show all commands
sfb --help

# Show command-specific help
sfb backtest --help
sfb sweep --help
sfb generate-bars --help
```

## Next Steps

- [Strategy Development Guide](strategies.md) - Learn to implement custom strategies
- [Alternative Bar Types](bar_types.md) - Deep dive into all 7 bar types
- [API Reference](../api/reference.md) - Complete Python API documentation
