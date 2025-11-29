# Strategy Development Guide

This guide explains how to implement trading strategies using the Simple Futures Backtester framework. You'll learn the BaseStrategy interface, see examples of the three built-in strategies, and create your own custom strategy.

## BaseStrategy Interface

All strategies must inherit from `BaseStrategy` and implement the `generate_signals()` method.

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

from simple_futures_backtester.config import StrategyConfig


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Attributes:
        config: StrategyConfig containing name and parameters dict.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize strategy with configuration.

        Args:
            config: StrategyConfig with strategy name and parameters.
        """
        self.config = config

    @abstractmethod
    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        """Generate trading signals from OHLCV data.

        Args:
            open_arr: Opening prices as float64 array.
            high_arr: High prices as float64 array.
            low_arr: Low prices as float64 array.
            close_arr: Closing prices as float64 array.
            volume_arr: Volume as int64 array.

        Returns:
            Signal array with values:
                1  = Long signal
               -1  = Short signal
                0  = Flat (no position)

            Array length must match close_arr length.
        """
        pass
```

### Signal Array Format

The `generate_signals()` method returns an `int32` numpy array with three possible values:

| Value | Meaning | Action |
|-------|---------|--------|
| `1` | Long signal | Enter or maintain long position |
| `-1` | Short signal | Enter or maintain short position |
| `0` | Flat signal | Exit any position or stay flat |

**Signal Types:**

- **STATE signals**: Signal persists while condition holds (e.g., Mean Reversion)
- **EVENT signals**: Signal only on transition bar (e.g., Breakout)

The backtest engine converts state signals to entry/exit events automatically.

## Built-in Strategy Examples

### 1. Momentum Strategy

Uses RSI and EMA crossover to identify trend momentum.

**Signal Logic:**
- Long: RSI > 50 AND fast_ema > slow_ema
- Short: RSI < 50 AND fast_ema < slow_ema
- Flat: Otherwise

**Implementation:**

```python
import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy, register_strategy


@register_strategy("momentum")
class MomentumStrategy(BaseStrategy):
    """RSI + EMA crossover momentum strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        # Extract parameters with defaults
        self.rsi_period: int = config.parameters.get("rsi_period", 14)
        self.fast_ema: int = config.parameters.get("fast_ema", 9)
        self.slow_ema: int = config.parameters.get("slow_ema", 21)

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        # Initialize signals to flat
        signals = np.zeros(len(close_arr), dtype=np.int32)

        # Convert to pandas Series for VectorBT
        close_series = pd.Series(close_arr)

        # Calculate RSI
        rsi = vbt.RSI.run(close_series, window=self.rsi_period).rsi.values

        # Calculate EMAs
        fast_ema = vbt.MA.run(close_series, window=self.fast_ema, ewm=True).ma.values
        slow_ema = vbt.MA.run(close_series, window=self.slow_ema, ewm=True).ma.values

        # Generate long signals
        long_condition = (rsi > 50) & (fast_ema > slow_ema)
        signals[long_condition] = 1

        # Generate short signals
        short_condition = (rsi < 50) & (fast_ema < slow_ema)
        signals[short_condition] = -1

        return signals
```

**YAML Configuration:**

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

### 2. Mean Reversion Strategy

Uses Bollinger Bands to identify oversold/overbought conditions.

**Signal Logic:**
- Long: close <= lower_band (oversold)
- Short: close >= upper_band (overbought)
- Flat: Price between bands

**Implementation:**

```python
import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy, register_strategy


@register_strategy("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.bb_period: int = config.parameters.get("bb_period", 20)
        self.bb_std: float = config.parameters.get("bb_std", 2.0)

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        signals = np.zeros(len(close_arr), dtype=np.int32)

        close_series = pd.Series(close_arr)

        # Calculate Bollinger Bands
        bb = vbt.BBANDS.run(close_series, window=self.bb_period, alpha=self.bb_std)
        upper_band = bb.upper.values
        lower_band = bb.lower.values

        # Long when oversold (price at lower band)
        long_condition = close_arr <= lower_band
        signals[long_condition] = 1

        # Short when overbought (price at upper band)
        short_condition = close_arr >= upper_band
        signals[short_condition] = -1

        return signals
```

**YAML Configuration:**

```yaml
strategy:
  name: mean_reversion
  parameters:
    bb_period: 20
    bb_std: 2.0

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

### 3. Breakout Strategy

Uses EMA crossover to identify trend breakouts.

**Signal Logic:**
- Long: Fast EMA crosses above slow EMA
- Short: Fast EMA crosses below slow EMA
- Flat: No crossover

**Implementation:**

```python
import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy, register_strategy


@register_strategy("breakout")
class BreakoutStrategy(BaseStrategy):
    """EMA crossover breakout strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.fast_period: int = config.parameters.get("fast_period", 10)
        self.slow_period: int = config.parameters.get("slow_period", 30)

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        signals = np.zeros(len(close_arr), dtype=np.int32)

        close_series = pd.Series(close_arr)

        # Calculate EMAs
        fast_ema = vbt.MA.run(close_series, window=self.fast_period, ewm=True).ma
        slow_ema = vbt.MA.run(close_series, window=self.slow_period, ewm=True).ma

        # Detect crossovers
        fast_above_slow = fast_ema > slow_ema
        fast_above_slow_prev = fast_above_slow.shift(1).fillna(False)

        # Long entry: crossed above
        crossed_above = (~fast_above_slow_prev) & fast_above_slow
        signals[crossed_above.values] = 1

        # Short entry: crossed below
        crossed_below = fast_above_slow_prev & (~fast_above_slow)
        signals[crossed_below.values] = -1

        return signals
```

**YAML Configuration:**

```yaml
strategy:
  name: breakout
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

## Creating a Custom Strategy

### Step 1: Create Strategy File

Create `simple_futures_backtester/strategy/examples/my_strategy.py`:

```python
"""My custom trading strategy.

Combines MACD with ATR-based volatility filter to identify
trending markets with sufficient volatility for profitable trades.

Signal Logic:
    Long (1):  MACD > Signal AND ATR > threshold
    Short (-1): MACD < Signal AND ATR > threshold
    Flat (0):  Low volatility environment

Parameters:
    macd_fast: MACD fast period (default: 12)
    macd_slow: MACD slow period (default: 26)
    macd_signal: MACD signal period (default: 9)
    atr_period: ATR lookback period (default: 14)
    atr_threshold: Minimum ATR for signals (default: 0.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy, register_strategy


@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    """MACD + ATR volatility filter strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        # MACD parameters
        self.macd_fast: int = config.parameters.get("macd_fast", 12)
        self.macd_slow: int = config.parameters.get("macd_slow", 26)
        self.macd_signal: int = config.parameters.get("macd_signal", 9)
        # ATR filter parameters
        self.atr_period: int = config.parameters.get("atr_period", 14)
        self.atr_threshold: float = config.parameters.get("atr_threshold", 0.5)

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        signals = np.zeros(len(close_arr), dtype=np.int32)

        # Convert to pandas for VectorBT
        close_series = pd.Series(close_arr)
        high_series = pd.Series(high_arr)
        low_series = pd.Series(low_arr)

        # Calculate MACD
        macd = vbt.MACD.run(
            close_series,
            fast_window=self.macd_fast,
            slow_window=self.macd_slow,
            signal_window=self.macd_signal,
        )
        macd_line = macd.macd.values
        signal_line = macd.signal.values

        # Calculate ATR for volatility filter
        atr = vbt.ATR.run(
            high_series,
            low_series,
            close_series,
            window=self.atr_period,
        ).atr.values

        # Volatility filter: only trade when ATR > threshold
        volatility_ok = atr > self.atr_threshold

        # Long: MACD above signal with sufficient volatility
        long_condition = (macd_line > signal_line) & volatility_ok
        signals[long_condition] = 1

        # Short: MACD below signal with sufficient volatility
        short_condition = (macd_line < signal_line) & volatility_ok
        signals[short_condition] = -1

        return signals
```

### Step 2: Register the Strategy

The `@register_strategy("my_strategy")` decorator automatically registers your strategy. To make it discoverable, add an import to `simple_futures_backtester/strategy/examples/__init__.py`:

```python
from simple_futures_backtester.strategy.examples.my_strategy import MyStrategy
```

### Step 3: Create Configuration

Create `configs/strategies/my_strategy.yaml`:

```yaml
strategy:
  name: my_strategy
  parameters:
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    atr_period: 14
    atr_threshold: 0.5

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

### Step 4: Run Backtest

```bash
sfb backtest \
    --strategy my_strategy \
    --config configs/strategies/my_strategy.yaml \
    --data examples/sample_data/es_1min_sample.csv
```

## Testing Your Strategy

### Unit Testing Signals

Create `tests/test_my_strategy.py`:

```python
import numpy as np
import pytest

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.examples.my_strategy import MyStrategy


def test_my_strategy_signal_values():
    """Test that signals only contain valid values."""
    config = StrategyConfig(
        name="my_strategy",
        parameters={"macd_fast": 12, "macd_slow": 26, "atr_threshold": 0.1},
    )
    strategy = MyStrategy(config)

    # Create synthetic data with trend
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    open_arr = close - 0.5
    high = close + 1
    low = close - 1
    volume = np.random.randint(1000, 10000, n)

    signals = strategy.generate_signals(
        open_arr.astype(np.float64),
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
        volume.astype(np.int64),
    )

    # Verify signal values
    assert signals.dtype == np.int32
    assert len(signals) == n
    assert set(np.unique(signals)).issubset({-1, 0, 1})


def test_my_strategy_warmup_period():
    """Test that signals are flat during warmup."""
    config = StrategyConfig(
        name="my_strategy",
        parameters={"macd_slow": 26, "atr_period": 14},
    )
    strategy = MyStrategy(config)

    # Short data: less than warmup period
    n = 20  # Less than macd_slow=26
    close = np.ones(n) * 100
    signals = strategy.generate_signals(
        close.astype(np.float64),
        (close + 1).astype(np.float64),
        (close - 1).astype(np.float64),
        close.astype(np.float64),
        np.ones(n, dtype=np.int64) * 1000,
    )

    # Most signals should be flat during warmup
    # (NaN indicators return False for comparisons)
    assert signals[0] == 0  # First bar always flat
```

Run tests:

```bash
pytest tests/test_my_strategy.py -v
```

### Validating with Known Data

Test your strategy against expected outcomes:

```python
def test_my_strategy_uptrend():
    """Test long signals in uptrend with high volatility."""
    config = StrategyConfig(
        name="my_strategy",
        parameters={"atr_threshold": 0.1},
    )
    strategy = MyStrategy(config)

    # Create clear uptrend with volatility
    n = 100
    close = np.linspace(100, 150, n)  # Steady uptrend
    high = close + 2  # High volatility
    low = close - 2
    open_arr = close - 0.5
    volume = np.ones(n, dtype=np.int64) * 1000

    signals = strategy.generate_signals(
        open_arr.astype(np.float64),
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
        volume,
    )

    # After warmup, should have long signals in uptrend
    post_warmup = signals[50:]  # Skip warmup period
    assert np.sum(post_warmup == 1) > np.sum(post_warmup == -1)
```

## Parameter Tuning with Sweep

Optimize your strategy parameters:

### Create Sweep Configuration

Create `configs/sweeps/my_strategy_sweep.yaml`:

```yaml
sweep:
  strategy: my_strategy
  parameters:
    macd_fast: [8, 12, 16]
    macd_slow: [20, 26, 32]
    macd_signal: [7, 9, 11]
    atr_threshold: [0.3, 0.5, 0.7]

backtest:
  initial_capital: 100000
  fees: 0.0001
  slippage: 0.0001
  size: 1
  size_type: fixed
  freq: 1D
```

### Run Sweep

```bash
sfb sweep \
    --strategy my_strategy \
    --sweep-config configs/sweeps/my_strategy_sweep.yaml \
    --data examples/sample_data/es_1min_sample.csv \
    --n-jobs -1
```

This tests all 81 combinations (3 x 3 x 3 x 3) in parallel.

## VectorBT Indicators

VectorBT provides many built-in indicators. Common ones:

| Indicator | Usage | Parameters |
|-----------|-------|------------|
| RSI | `vbt.RSI.run(close, window=14).rsi` | window |
| MA/EMA | `vbt.MA.run(close, window=20, ewm=True).ma` | window, ewm |
| MACD | `vbt.MACD.run(close, fast_window, slow_window, signal_window)` | 3 windows |
| Bollinger Bands | `vbt.BBANDS.run(close, window=20, alpha=2.0)` | window, alpha |
| ATR | `vbt.ATR.run(high, low, close, window=14).atr` | window |
| Stochastic | `vbt.STOCH.run(high, low, close, k_window, d_window)` | k, d windows |

**Example: Using Multiple Indicators**

```python
import vectorbt as vbt

# RSI for momentum
rsi = vbt.RSI.run(close_series, window=14).rsi.values

# Bollinger Bands for volatility
bb = vbt.BBANDS.run(close_series, window=20, alpha=2.0)
upper = bb.upper.values
lower = bb.lower.values

# ATR for volatility filtering
atr = vbt.ATR.run(high_series, low_series, close_series, window=14).atr.values

# Combine into signals
long_signals = (rsi > 50) & (close_arr < lower) & (atr > threshold)
```

## Best Practices

### 1. Handle Warmup Period

Indicators need time to calculate. During warmup, they return NaN values which evaluate to False in boolean comparisons.

```python
# Warmup period = max indicator period
warmup = max(self.rsi_period, self.slow_ema)

# Alternatively, explicitly handle NaN
rsi = vbt.RSI.run(close_series, window=self.rsi_period).rsi.values
valid = ~np.isnan(rsi)
signals[valid & (rsi > 70)] = -1
```

### 2. Avoid Look-Ahead Bias

Never use future data to make current signals:

```python
# WRONG: Uses future close prices
future_avg = np.convolve(close_arr, np.ones(10)/10, mode='full')[:len(close_arr)]

# CORRECT: Use only past data
past_avg = pd.Series(close_arr).rolling(10).mean().values
```

### 3. Use Efficient NumPy Operations

```python
# WRONG: Slow Python loop
for i in range(len(close_arr)):
    if close_arr[i] > upper_band[i]:
        signals[i] = -1

# CORRECT: Vectorized operation
signals[close_arr > upper_band] = -1
```

### 4. Document Signal Logic

Always document your signal logic clearly:

```python
"""
Signal Logic:
    Long (1):  RSI > 50 AND price > 20-day MA
    Short (-1): RSI < 50 AND price < 20-day MA
    Flat (0):  RSI between 45-55 (neutral zone)
"""
```

## Next Steps

- [Alternative Bar Types](bar_types.md) - Use non-time-based bars with your strategy
- [API Reference](../api/reference.md) - Complete Python API documentation
- [Quick Start](quickstart.md) - Back to basics
