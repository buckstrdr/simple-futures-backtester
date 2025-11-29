# Alternative Bar Types Guide

This guide documents all 7 alternative bar types available in Simple Futures Backtester. Alternative bars sample market data differently than traditional time-based bars, often providing better signals for specific trading strategies.

## Why Alternative Bars?

Traditional time-based bars (1-minute, 5-minute, daily) have limitations:

1. **Unequal Information**: A 1-minute bar during market open contains more information than during lunch hour
2. **Noise Amplification**: Low-activity periods create small, noisy bars
3. **Time Dependence**: Bar count depends on time elapsed, not market activity

Alternative bars address these issues by sampling based on:
- **Price movement** (Renko, Range bars)
- **Volume/activity** (Volume, Dollar, Tick bars)
- **Order flow** (Tick Imbalance, Volume Imbalance bars)

## Bar Types Overview

| Bar Type | Closes When | Best For | Key Parameter |
|----------|-------------|----------|---------------|
| **Renko** | Price moves by brick_size | Trend following | `brick_size` |
| **Range** | High-Low range exceeds threshold | Volatility trading | `range_size` |
| **Tick** | Fixed number of source bars | Downsampling | `tick_threshold` |
| **Volume** | Cumulative volume exceeds threshold | Activity-normalized | `volume_threshold` |
| **Dollar** | Cumulative dollar volume exceeds threshold | Value-normalized | `dollar_threshold` |
| **Tick Imbalance** | Signed tick imbalance exceeds threshold | Order flow analysis | `threshold` |
| **Volume Imbalance** | Volume-weighted imbalance exceeds threshold | Institutional flow | `threshold` |

## Renko Bars

### Definition

Renko bars ignore time and form based purely on price movement magnitude. A new brick forms when price moves by a fixed amount (brick_size) or more. Reversals require 2x the brick size.

### Algorithm

1. Start with initial close price as base
2. If price moves >= brick_size in current direction, create brick
3. If price moves >= 2 x brick_size in opposite direction, reverse and create brick
4. Repeat for all price updates

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `brick_size` | float | Fixed brick size in price units (e.g., 10 for ES = 10 points) |
| `atr_length` | int | If > 0, uses ATR for dynamic brick sizing |
| `atr_values` | array | Pre-computed ATR values for dynamic sizing |

### Use Case

**When to use Renko:**
- Trend-following strategies in choppy markets
- Filtering out small price fluctuations
- Visual clarity for support/resistance levels
- Markets with significant noise during consolidation

**When NOT to use Renko:**
- Scalping or high-frequency strategies
- When exact timing matters
- Mean reversion strategies (Renko smooths reversals)

### Tuning Guidance

- **Fixed brick_size**: Set to 1-2x average bar range for your timeframe
- **ATR-based**: Use `atr_length=14` for adaptive sizing that responds to volatility
- **Futures guidance**: For ES (E-mini S&P), try brick_size = 2-5 points

### Code Example

```python
from simple_futures_backtester.bars import generate_renko_bars_series
from simple_futures_backtester.data import load_csv
import numpy as np

# Load OHLCV data
df = load_csv("examples/sample_data/es_1min_sample.csv")

# Generate Renko bars with 5-point bricks
renko_bars = generate_renko_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    brick_size=5.0,
)

print(f"Generated {len(renko_bars)} Renko bars from {len(df)} source bars")
print(f"Compression ratio: {len(df)/len(renko_bars):.1f}:1")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type renko \
    --param 5.0 \
    --data examples/sample_data/es_1min_sample.csv \
    --output renko_bars.csv
```

### Comparison to Time Bars

| Aspect | Time Bars | Renko Bars |
|--------|-----------|------------|
| Time axis | Preserved | Eliminated |
| Noise filtering | None | High (small moves ignored) |
| Trend visibility | Can be obscured | Very clear |
| Bar count | Fixed by time | Variable by volatility |
| Gaps | Preserved | Filled with synthetic bricks |

## Range Bars

### Definition

Range bars form when the cumulative high-low range since the last bar exceeds a threshold. Each bar represents a fixed amount of price volatility, regardless of time.

### Algorithm

1. Track running high and low from bar start
2. When (running_high - running_low) >= range_size, close bar
3. Reset running high/low to current values
4. Repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `range_size` | float | Range threshold in price units |

### Use Case

**When to use Range bars:**
- Volatility-focused strategies
- Filtering low-volatility periods
- Normalizing bars across different market conditions
- Identifying volatility breakouts

**When NOT to use Range bars:**
- When time-of-day matters
- Very short-term strategies
- When you need consistent bar counts

### Tuning Guidance

- Set range_size to ~1.5x average true range (ATR) of your target timeframe
- For ES (E-mini S&P): Try 2-4 points
- Higher range_size = fewer bars, smoother trends

### Code Example

```python
from simple_futures_backtester.bars import generate_range_bars_series
import numpy as np

# Generate Range bars with 3-point range threshold
range_bars = generate_range_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    range_size=3.0,
)

print(f"Generated {len(range_bars)} Range bars")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type range \
    --param 3.0 \
    --data examples/sample_data/es_1min_sample.csv
```

## Tick Bars

### Definition

Tick bars aggregate a fixed number of source bars (or trades) into each output bar. This downsamples high-frequency data uniformly.

### Algorithm

1. Accumulate `tick_threshold` source bars
2. Create output bar with aggregated OHLCV
3. Reset counter and repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tick_threshold` | int | Number of source bars per output bar |

### Use Case

**When to use Tick bars:**
- Downsampling high-frequency tick data
- Creating uniform-activity bars from trades
- Normalizing bar count across sessions
- When each "tick" represents equal information

**When NOT to use Tick bars:**
- When time matters for your strategy
- With already aggregated (e.g., 1-minute) data
- When you need volume information per bar

### Tuning Guidance

- For tick data: 100-1000 ticks per bar depending on activity
- For 1-minute data: 5-10 bars = roughly hourly grouping
- Match tick_threshold to your strategy's holding period

### Code Example

```python
from simple_futures_backtester.bars import generate_tick_bars_series
import numpy as np

# Aggregate every 10 source bars into one tick bar
tick_bars = generate_tick_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    tick_threshold=10,
)

print(f"Generated {len(tick_bars)} Tick bars from {len(df)} source bars")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type tick \
    --param 10 \
    --data examples/sample_data/es_1min_sample.csv
```

## Volume Bars

### Definition

Volume bars close when cumulative volume reaches a threshold. Each bar represents a fixed amount of trading activity, normalizing for market participation.

### Algorithm

1. Accumulate volume from source bars
2. When cumulative_volume >= volume_threshold, close bar
3. Reset cumulative volume to zero
4. Repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `volume_threshold` | int | Volume threshold for closing bars |

### Use Case

**When to use Volume bars:**
- Activity-normalized strategies
- When volume is a key signal
- Markets with variable activity (pre-market vs regular hours)
- Institutional flow analysis

**When NOT to use Volume bars:**
- Low-volume instruments
- When time-based patterns matter
- Very thin markets with erratic volume

### Tuning Guidance

- Set threshold to average volume per desired "bar equivalent"
- For ES: Average daily volume / desired_bars_per_day
- Start with median volume per source bar x 10-50

### Code Example

```python
from simple_futures_backtester.bars import generate_volume_bars_series
import numpy as np

# Generate Volume bars with 10,000 volume threshold
volume_bars = generate_volume_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    volume_threshold=10000,
)

print(f"Generated {len(volume_bars)} Volume bars")
print(f"Average volume per bar: {volume_bars.volume.mean():.0f}")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type volume \
    --param 10000 \
    --data examples/sample_data/es_1min_sample.csv
```

## Dollar Bars

### Definition

Dollar bars close when cumulative dollar volume (price x volume) exceeds a threshold. This normalizes for both price level and activity.

### Algorithm

1. Calculate dollar_volume = close_price x volume for each source bar
2. Accumulate dollar volume
3. When cumulative >= dollar_threshold, close bar
4. Reset and repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dollar_threshold` | float | Dollar volume threshold for closing bars |

### Use Case

**When to use Dollar bars:**
- Comparing across instruments with different prices
- Value-weighted activity normalization
- Cross-asset strategies
- When both price and volume matter

**When NOT to use Dollar bars:**
- Single-instrument strategies where volume bars suffice
- When price changes significantly during session
- Very low-priced instruments

### Tuning Guidance

- Set threshold = average_price x volume_per_bar
- For ES at 4500: Try 45,000,000 - 450,000,000
- Adjust based on desired compression ratio

### Code Example

```python
from simple_futures_backtester.bars import generate_dollar_bars_series
import numpy as np

# Generate Dollar bars with $50M threshold
dollar_bars = generate_dollar_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    dollar_threshold=50_000_000.0,
)

print(f"Generated {len(dollar_bars)} Dollar bars")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type dollar \
    --param 50000000 \
    --data examples/sample_data/es_1min_sample.csv
```

## Tick Imbalance Bars

### Definition

Tick imbalance bars track the signed direction of price changes (uptick/downtick). A new bar forms when the absolute imbalance exceeds a threshold, capturing order flow imbalances.

### Algorithm

1. For each source bar, determine tick direction: +1 (uptick) or -1 (downtick)
2. Accumulate signed tick count
3. When |cumulative_imbalance| >= threshold, close bar
4. Reset imbalance and repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | int | Absolute imbalance threshold |

### Use Case

**When to use Tick Imbalance bars:**
- Microstructure analysis
- Order flow strategies
- Detecting buying vs selling pressure
- High-frequency pattern detection

**When NOT to use Tick Imbalance bars:**
- Swing trading (too granular)
- Already aggregated data (need tick-level)
- When volume matters more than direction

### Tuning Guidance

- Threshold should be ~5-20% of typical bar tick count
- Lower threshold = more bars, faster reaction
- Higher threshold = fewer bars, stronger signal

### Code Example

```python
from simple_futures_backtester.bars import generate_tick_imbalance_bars_series
import numpy as np

# Generate Tick Imbalance bars with threshold of 50
tick_imb_bars = generate_tick_imbalance_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    threshold=50,
)

print(f"Generated {len(tick_imb_bars)} Tick Imbalance bars")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type tick_imbalance \
    --param 50 \
    --data examples/sample_data/es_1min_sample.csv
```

## Volume Imbalance Bars

### Definition

Volume imbalance bars weight tick direction by volume, capturing volume-weighted order flow. Bars close when the absolute volume-weighted imbalance exceeds a threshold.

### Algorithm

1. For each source bar:
   - If uptick: imbalance += volume
   - If downtick: imbalance -= volume
2. When |cumulative_imbalance| >= threshold, close bar
3. Reset and repeat

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | int | Absolute volume imbalance threshold |

### Use Case

**When to use Volume Imbalance bars:**
- Institutional flow detection
- Volume-weighted order flow analysis
- Large trader activity signals
- When volume confirms direction matters

**When NOT to use Volume Imbalance bars:**
- Low-volume markets
- When volume is unreliable/sparse
- Trend-following without volume context

### Tuning Guidance

- Threshold ~10-50% of typical bar volume
- Similar to volume bars but directional
- Start with median bar volume x 10

### Code Example

```python
from simple_futures_backtester.bars import generate_volume_imbalance_bars_series
import numpy as np

# Generate Volume Imbalance bars with threshold of 50,000
vol_imb_bars = generate_volume_imbalance_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    threshold=50000,
)

print(f"Generated {len(vol_imb_bars)} Volume Imbalance bars")
```

**CLI:**

```bash
sfb generate-bars \
    --bar-type volume_imbalance \
    --param 50000 \
    --data examples/sample_data/es_1min_sample.csv
```

## Choosing the Right Bar Type

Use this decision matrix to select the best bar type for your strategy:

| If Your Strategy... | Recommended Bar Type | Why |
|---------------------|---------------------|-----|
| Follows trends | Renko | Filters noise, clear trends |
| Trades volatility breakouts | Range | Captures volatility events |
| Needs equal activity sampling | Volume | Normalizes participation |
| Works across multiple instruments | Dollar | Price-normalized activity |
| Analyzes order flow | Tick/Volume Imbalance | Directional flow detection |
| Downsamples high-frequency data | Tick | Uniform aggregation |

### Strategy-Bar Type Pairings

| Strategy Type | Primary Bar | Secondary Option |
|---------------|-------------|------------------|
| Trend Following | Renko | Range |
| Mean Reversion | Range | Volume |
| Momentum | Volume | Tick |
| Scalping | Tick | Tick Imbalance |
| Institutional Flow | Volume Imbalance | Dollar |
| Breakout | Range | Renko |

## Using Alternative Bars with Strategies

### Complete Workflow

```python
from simple_futures_backtester.data import load_csv
from simple_futures_backtester.bars import generate_renko_bars_series
from simple_futures_backtester.strategy import get_strategy
from simple_futures_backtester.config import StrategyConfig, BacktestConfig
from simple_futures_backtester.backtest import BacktestEngine
import numpy as np

# 1. Load source data
df = load_csv("examples/sample_data/es_1min_sample.csv")

# 2. Generate alternative bars
renko = generate_renko_bars_series(
    open_arr=df["open"].values.astype(np.float64),
    high_arr=df["high"].values.astype(np.float64),
    low_arr=df["low"].values.astype(np.float64),
    close_arr=df["close"].values.astype(np.float64),
    volume_arr=df["volume"].values.astype(np.int64),
    brick_size=5.0,
)

# 3. Create strategy
StrategyClass = get_strategy("momentum")
config = StrategyConfig(
    name="momentum",
    parameters={"rsi_period": 14, "fast_ema": 9, "slow_ema": 21},
)
strategy = StrategyClass(config)

# 4. Generate signals on alternative bars
signals = strategy.generate_signals(
    renko.open,
    renko.high,
    renko.low,
    renko.close,
    renko.volume,
)

# 5. Run backtest
engine = BacktestEngine()
backtest_config = BacktestConfig(initial_capital=100000.0)
result = engine.run(renko.close, signals, backtest_config)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
print(f"Number of Trades: {result.n_trades}")
```

## Performance Notes

All bar generators are JIT-compiled with Numba for high performance:

| Operation | Target | Typical Performance |
|-----------|--------|---------------------|
| Renko generation | 1M+ rows/sec | 2-3M rows/sec |
| Range generation | 1M+ rows/sec | 2-3M rows/sec |
| Volume generation | 1M+ rows/sec | 2-3M rows/sec |
| Imbalance bars | 1M+ rows/sec | 1.5-2M rows/sec |

First call includes JIT compilation overhead (~1-2 seconds). Subsequent calls are fast.

## Next Steps

- [Strategy Development](strategies.md) - Use alternative bars with custom strategies
- [API Reference](../api/reference.md) - Complete function signatures
- [Quick Start](quickstart.md) - Back to basics
