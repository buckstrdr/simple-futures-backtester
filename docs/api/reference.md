# API Reference

Complete Python API documentation for Simple Futures Backtester.

## Table of Contents

- [Configuration API](#configuration-api)
- [Data Loading API](#data-loading-api)
- [Bar Generators API](#bar-generators-api)
- [Strategy API](#strategy-api)
- [Backtest API](#backtest-api)
- [Output API](#output-api)
- [Extensions API](#extensions-api)

---

## Configuration API

**Module:** `simple_futures_backtester.config`

### StrategyConfig

```python
@dataclass
class StrategyConfig:
    """Configuration for a trading strategy.

    Attributes:
        name: Strategy name (used for registry lookup).
        parameters: Dictionary of strategy-specific parameters.
    """
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
from simple_futures_backtester.config import StrategyConfig

config = StrategyConfig(
    name="momentum",
    parameters={
        "rsi_period": 14,
        "fast_ema": 9,
        "slow_ema": 21,
    },
)
```

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    """Configuration for backtest execution.

    Attributes:
        initial_capital: Starting capital in dollars.
        fees: Transaction fees as decimal (0.0001 = 0.01%).
        slippage: Slippage as decimal (0.0001 = 0.01%).
        size: Position size (contracts or shares).
        size_type: How to interpret size ("fixed", "percent", "target").
        freq: Bar frequency for portfolio ("1D", "1H", "1T").
    """
    initial_capital: float = 100_000.0
    fees: float = 0.0001
    slippage: float = 0.0001
    size: int = 1
    size_type: str = "fixed"
    freq: str = "1D"
```

**Size Types:**

| Size Type | Description |
|-----------|-------------|
| `"fixed"` | Fixed number of contracts/shares |
| `"percent"` | Percentage of available capital |
| `"target"` | Target position size |

**Example:**

```python
from simple_futures_backtester.config import BacktestConfig

config = BacktestConfig(
    initial_capital=50000.0,
    fees=0.001,     # 0.1% transaction fee
    slippage=0.0005, # 0.05% slippage
    size=2,         # 2 contracts
    size_type="fixed",
    freq="1D",
)
```

### SweepConfig

```python
@dataclass
class SweepConfig:
    """Configuration for parameter sweep optimization.

    Attributes:
        strategy: Strategy name to optimize.
        parameters: Dictionary mapping param name to list of values to test.
        backtest_overrides: Optional overrides for BacktestConfig fields.
    """
    strategy: str
    parameters: dict[str, list[Any]]
    backtest_overrides: dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
from simple_futures_backtester.config import SweepConfig

sweep = SweepConfig(
    strategy="momentum",
    parameters={
        "rsi_period": [10, 14, 20],
        "fast_ema": [5, 9, 12],
        "slow_ema": [21, 30, 50],
    },
    backtest_overrides={
        "initial_capital": 50000.0,
        "fees": 0.002,
    },
)
```

### LoadedConfig

```python
@dataclass
class LoadedConfig:
    """Container for a loaded configuration with its hash.

    Attributes:
        strategy: The loaded StrategyConfig (if applicable).
        backtest: The loaded BacktestConfig (always present with defaults).
        sweep: The loaded SweepConfig (if applicable).
        config_hash: SHA256 hash of the source YAML for reproducibility.
        source_path: Path to the source YAML file.
    """
    strategy: StrategyConfig | None = None
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sweep: SweepConfig | None = None
    config_hash: str = ""
    source_path: str = ""
```

### load_config

```python
def load_config(config_path: str | Path) -> LoadedConfig:
    """Load configuration from a YAML file with environment overrides.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        LoadedConfig containing the parsed configuration and its hash.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If required fields are missing.
    """
```

**Example:**

```python
from simple_futures_backtester.config import load_config

loaded = load_config("configs/strategies/momentum.yaml")
print(loaded.strategy.name)  # "momentum"
print(loaded.backtest.initial_capital)  # 100000.0
print(loaded.config_hash[:16])  # First 16 chars of hash
```

### load_strategy_config

```python
def load_strategy_config(config_path: str | Path) -> tuple[StrategyConfig, BacktestConfig, str]:
    """Convenience function to load a strategy configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (StrategyConfig, BacktestConfig, config_hash).

    Raises:
        ValueError: If the config file does not contain a strategy section.
    """
```

### load_sweep_config

```python
def load_sweep_config(config_path: str | Path) -> tuple[SweepConfig, BacktestConfig, str]:
    """Convenience function to load a sweep configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (SweepConfig, BacktestConfig, config_hash).

    Raises:
        ValueError: If the config file does not contain a sweep section.
    """
```

### compute_config_hash

```python
def compute_config_hash(yaml_content: str) -> str:
    """Compute SHA256 hash of YAML content for reproducibility tracking.

    Args:
        yaml_content: Raw YAML file content as a string.

    Returns:
        Hexadecimal SHA256 hash of the normalized content.
    """
```

**Environment Variable Overrides:**

BacktestConfig fields can be overridden via environment variables:

| Environment Variable | Field |
|---------------------|-------|
| `SFB_BACKTEST_INITIAL_CAPITAL` or `SFB_CAPITAL` | `initial_capital` |
| `SFB_BACKTEST_FEES` or `SFB_FEES` | `fees` |
| `SFB_BACKTEST_SLIPPAGE` or `SFB_SLIPPAGE` | `slippage` |
| `SFB_BACKTEST_SIZE` or `SFB_SIZE` | `size` |
| `SFB_BACKTEST_SIZE_TYPE` | `size_type` |
| `SFB_BACKTEST_FREQ` | `freq` |

---

## Data Loading API

**Module:** `simple_futures_backtester.data`

### load_csv

```python
def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV data from CSV file.

    Validates schema and normalizes column names to lowercase.
    Parses datetime column to UTC timezone.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume.
        Types: datetime64[ns, UTC], float64, float64, float64, float64, int64.

    Raises:
        FileNotFoundError: If file doesn't exist.
        DataLoadError: If schema validation fails.
    """
```

**Required CSV Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | parsed | Bar timestamp |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | int64 | Trading volume |

**Example:**

```python
from simple_futures_backtester.data import load_csv

df = load_csv("examples/sample_data/es_1min_sample.csv")
print(df.dtypes)
# datetime    datetime64[ns, UTC]
# open                    float64
# high                    float64
# low                     float64
# close                   float64
# volume                    int64
```

### load_parquet

```python
def load_parquet(file_path: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV data from Parquet file.

    Same schema validation as load_csv.

    Args:
        file_path: Path to Parquet file.

    Returns:
        DataFrame with validated OHLCV schema.

    Raises:
        FileNotFoundError: If file doesn't exist.
        DataLoadError: If schema validation fails.
    """
```

### DataLoadError

```python
class DataLoadError(Exception):
    """Raised when data loading fails due to schema violations or file errors."""
```

---

## Bar Generators API

**Module:** `simple_futures_backtester.bars`

### BarSeries

```python
@dataclass
class BarSeries:
    """Container for generated bar data.

    Attributes:
        type: Bar type name ("renko", "range", etc.).
        parameters: Dict of generation parameters used.
        open: Opening prices as float64 array.
        high: High prices as float64 array.
        low: Low prices as float64 array.
        close: Closing prices as float64 array.
        volume: Aggregated volume as int64 array.
        index_map: Source row index where each bar completed.
    """
    type: str
    parameters: dict[str, Any]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.int64]
    index_map: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.close)
```

### generate_renko_bars_series

```python
def generate_renko_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    brick_size: float | None = None,
    atr_length: int = 0,
    atr_values: NDArray[np.float64] | None = None,
) -> BarSeries:
    """Generate Renko bars from OHLCV data.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        brick_size: Fixed brick size. Required if atr_values not provided.
        atr_length: ATR lookback for dynamic sizing. Use with atr_values.
        atr_values: Pre-computed ATR for dynamic brick sizing.

    Returns:
        BarSeries with Renko bars.

    Raises:
        ValueError: If brick_size invalid or not provided.
    """
```

### generate_range_bars_series

```python
def generate_range_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    range_size: float,
) -> BarSeries:
    """Generate Range bars from OHLCV data.

    Closes bar when cumulative high-low range exceeds range_size.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        range_size: Range threshold in price units.

    Returns:
        BarSeries with Range bars.

    Raises:
        ValueError: If range_size <= 0.
    """
```

### generate_tick_bars_series

```python
def generate_tick_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    tick_threshold: int,
) -> BarSeries:
    """Generate Tick bars from OHLCV data.

    Aggregates fixed number of source bars per output bar.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        tick_threshold: Number of source bars per output bar.

    Returns:
        BarSeries with Tick bars.

    Raises:
        ValueError: If tick_threshold <= 0.
    """
```

### generate_volume_bars_series

```python
def generate_volume_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    volume_threshold: int,
) -> BarSeries:
    """Generate Volume bars from OHLCV data.

    Closes bar when cumulative volume exceeds volume_threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        volume_threshold: Volume threshold for closing bars.

    Returns:
        BarSeries with Volume bars.

    Raises:
        ValueError: If volume_threshold <= 0.
    """
```

### generate_dollar_bars_series

```python
def generate_dollar_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    dollar_threshold: float,
) -> BarSeries:
    """Generate Dollar bars from OHLCV data.

    Closes bar when cumulative dollar volume (price x volume) exceeds threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        dollar_threshold: Dollar volume threshold.

    Returns:
        BarSeries with Dollar bars.

    Raises:
        ValueError: If dollar_threshold <= 0.
    """
```

### generate_tick_imbalance_bars_series

```python
def generate_tick_imbalance_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    threshold: int,
) -> BarSeries:
    """Generate Tick Imbalance bars from OHLCV data.

    Tracks signed tick direction, closes when |imbalance| exceeds threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        threshold: Absolute imbalance threshold.

    Returns:
        BarSeries with Tick Imbalance bars.

    Raises:
        ValueError: If threshold <= 0.
    """
```

### generate_volume_imbalance_bars_series

```python
def generate_volume_imbalance_bars_series(
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    threshold: int,
) -> BarSeries:
    """Generate Volume Imbalance bars from OHLCV data.

    Weights tick direction by volume, closes when |imbalance| exceeds threshold.

    Args:
        open_arr: Opening prices.
        high_arr: High prices.
        low_arr: Low prices.
        close_arr: Closing prices.
        volume_arr: Volume data.
        threshold: Absolute volume imbalance threshold.

    Returns:
        BarSeries with Volume Imbalance bars.

    Raises:
        ValueError: If threshold <= 0.
    """
```

### Bar Type Registry Functions

```python
def register_bar_type(name: str, generator_func: BarGeneratorFunc) -> None:
    """Register a bar generator function with the factory.

    Args:
        name: Bar type identifier (e.g., "renko", "range").
        generator_func: Function that generates bars from OHLCV data.

    Raises:
        ValueError: If name is empty or generator_func is not callable.
    """

def get_bar_generator(name: str) -> BarGeneratorFunc:
    """Retrieve a registered bar generator by name.

    Args:
        name: Bar type identifier.

    Returns:
        The registered bar generator function.

    Raises:
        KeyError: If bar type is not registered.
    """

def list_bar_types() -> list[str]:
    """List all registered bar types.

    Returns:
        List of bar type identifiers in alphabetical order.
    """

def unregister_bar_type(name: str) -> bool:
    """Unregister a bar generator by name.

    Args:
        name: Bar type identifier to remove.

    Returns:
        True if the bar type was removed, False if it wasn't registered.
    """

def clear_bar_registry() -> None:
    """Clear all registered bar types. Useful for testing."""
```

**Example:**

```python
from simple_futures_backtester.bars import get_bar_generator, list_bar_types

# List available bar types
print(list_bar_types())  # ['dollar', 'range', 'renko', 'tick', ...]

# Get and use a generator
generator = get_bar_generator("renko")
bars = generator(open_arr, high_arr, low_arr, close_arr, volume_arr, brick_size=10)
print(f"Generated {len(bars)} Renko bars")
```

---

## Strategy API

**Module:** `simple_futures_backtester.strategy`

### Signal

```python
@dataclass
class Signal:
    """Individual trading signal specification.

    Attributes:
        direction: Signal direction (-1 for short, 0 for flat, 1 for long).
        size: Position size multiplier (default=1.0).
    """
    direction: int
    size: float = 1.0
```

**Example:**

```python
from simple_futures_backtester.strategy.base import Signal

signal = Signal(direction=1, size=1.0)   # Standard long
signal = Signal(direction=-1, size=0.5)  # Half-size short
signal = Signal(direction=0)              # Flat
```

### BaseStrategy

```python
class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Attributes:
        config: StrategyConfig containing name and parameters.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize strategy with configuration."""
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

        Returns:
            Signal array: 1 (long), -1 (short), 0 (flat).
        """
        pass
```

### register_strategy

```python
def register_strategy(name: str, strategy_class: type[BaseStrategy]) -> None:
    """Register a strategy class with the factory.

    Args:
        name: Strategy identifier (e.g., "momentum", "mean_reversion").
        strategy_class: Strategy class extending BaseStrategy.

    Raises:
        ValueError: If name is empty or strategy_class is not a BaseStrategy subclass.

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def generate_signals(self, ...):
        ...         return np.zeros(len(close_arr), dtype=np.int32)
        >>> register_strategy("my_strategy", MyStrategy)
    """
```

### get_strategy

```python
def get_strategy(name: str) -> type[BaseStrategy]:
    """Retrieve a registered strategy class by name.

    Args:
        name: Strategy identifier.

    Returns:
        Strategy class (NOT an instance).

    Raises:
        KeyError: If strategy is not registered.
    """
```

### list_strategies

```python
def list_strategies() -> list[str]:
    """List all registered strategies.

    Returns:
        List of strategy identifiers in alphabetical order.
    """
```

### unregister_strategy

```python
def unregister_strategy(name: str) -> bool:
    """Unregister a strategy by name.

    Args:
        name: Strategy identifier to remove.

    Returns:
        True if the strategy was removed, False if it wasn't registered.
    """
```

### clear_strategy_registry

```python
def clear_strategy_registry() -> None:
    """Clear all registered strategies. Useful for testing."""
```

**Example:**

```python
from simple_futures_backtester.strategy import get_strategy, list_strategies
from simple_futures_backtester.config import StrategyConfig

# List available strategies
print(list_strategies())  # ['breakout', 'mean_reversion', 'momentum']

# Get and instantiate a strategy
StrategyClass = get_strategy("momentum")
config = StrategyConfig(name="momentum", parameters={"rsi_period": 14})
strategy = StrategyClass(config)

# Generate signals
signals = strategy.generate_signals(open_arr, high_arr, low_arr, close_arr, volume_arr)
```

### Built-in Strategies

| Name | Class | Description |
|------|-------|-------------|
| `"momentum"` | `MomentumStrategy` | RSI + EMA crossover |
| `"mean_reversion"` | `MeanReversionStrategy` | Bollinger Bands |
| `"breakout"` | `BreakoutStrategy` | EMA crossover |

---

## Backtest API

**Module:** `simple_futures_backtester.backtest`

### BacktestEngine

```python
class BacktestEngine:
    """Backtesting engine using VectorBT Portfolio.

    Stateless engine that can be reused for multiple backtests.
    """

    def run(
        self,
        close: NDArray[np.float64],
        signals: NDArray[np.int32],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run backtest on signal array.

        Converts signal states to entry/exit events and runs
        VectorBT portfolio simulation.

        Args:
            close: Closing prices.
            signals: Signal array (-1, 0, 1).
            config: BacktestConfig with capital, fees, etc.

        Returns:
            BacktestResult with all metrics.

        Raises:
            ValueError: If close and signals length mismatch.
            ValueError: If signals contain invalid values.
        """
```

**Example:**

```python
from simple_futures_backtester.backtest import BacktestEngine
from simple_futures_backtester.config import BacktestConfig
import numpy as np

engine = BacktestEngine()
config = BacktestConfig(initial_capital=100000.0)

# signals from strategy.generate_signals()
result = engine.run(close_prices, signals, config)

print(f"Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.4f}")
```

### BacktestResult

```python
@dataclass
class BacktestResult:
    """Results from a backtest run.

    Scalar Metrics:
        total_return: Total return as decimal (0.15 = 15%).
        sharpe_ratio: Risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Maximum drawdown as decimal.
        win_rate: Winning trade percentage.
        profit_factor: Gross profits / Gross losses.
        n_trades: Total closed trades.
        avg_trade: Average trade profit/loss.

    Time-Series Data:
        equity_curve: Portfolio value at each bar (float64 array).
        drawdown_curve: Drawdown percentage at each bar (float64 array).
        trades: DataFrame with trade details.

    Metadata:
        config_hash: SHA256 hash of BacktestConfig.
        timestamp: ISO 8601 timestamp of execution.
    """
```

### ParameterSweep

```python
class ParameterSweep:
    """Brute-force grid search optimizer.

    Args:
        n_jobs: Parallel workers.
            1 = Sequential (debugging)
            -1 = All cores
            >1 = Specified workers
    """

    def __init__(self, n_jobs: int = 1) -> None: ...

    def run(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
        sweep_config: SweepConfig,
        base_backtest_config: BacktestConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SweepResult:
        """Execute grid search over all parameter combinations.

        Args:
            open_arr: Opening prices.
            high_arr: High prices.
            low_arr: Low prices.
            close_arr: Closing prices.
            volume_arr: Volume data.
            sweep_config: SweepConfig with parameter grid.
            base_backtest_config: Optional base config.
            progress_callback: Optional (current, total) callback.

        Returns:
            SweepResult with best params and all results.
        """
```

**Example:**

```python
from simple_futures_backtester.backtest import ParameterSweep
from simple_futures_backtester.config import SweepConfig

sweep_config = SweepConfig(
    strategy="momentum",
    parameters={
        "rsi_period": [10, 14, 20],
        "fast_ema": [5, 9],
    },
)

sweeper = ParameterSweep(n_jobs=-1)  # Use all cores
result = sweeper.run(open_arr, high_arr, low_arr, close_arr, volume_arr, sweep_config)

print(f"Best params: {result.best_params}")
print(f"Best Sharpe: {result.best_sharpe:.4f}")
```

### SweepResult

```python
@dataclass
class SweepResult:
    """Results from parameter sweep.

    Attributes:
        best_params: Parameter dict with highest Sharpe ratio.
        best_sharpe: Best Sharpe ratio found.
        all_results: List of (params_dict, BacktestResult) sorted by Sharpe.
    """
    best_params: dict[str, Any]
    best_sharpe: float
    all_results: list[tuple[dict[str, Any], BacktestResult]]
```

---

## Output API

**Module:** `simple_futures_backtester.output`

### ReportGenerator

```python
class ReportGenerator:
    """Generates formatted reports from backtest/sweep results."""

    @staticmethod
    def generate_text_report(
        result: BacktestResult | SweepResult,
        top_n: int = 10,
    ) -> str:
        """Generate Rich-formatted text report.

        Args:
            result: BacktestResult or SweepResult.
            top_n: For SweepResult, number of top combinations.

        Returns:
            String with Rich markup for console.print().
        """

    @staticmethod
    def generate_json_report(
        result: BacktestResult | SweepResult,
        strategy_name: str = "unknown",
    ) -> dict[str, Any]:
        """Generate JSON-serializable report.

        Args:
            result: BacktestResult or SweepResult.
            strategy_name: Strategy name for metadata.

        Returns:
            Dictionary for json.dumps().
        """
```

**Example:**

```python
from simple_futures_backtester.output import ReportGenerator
from rich.console import Console
import json

# Rich console output
text_report = ReportGenerator.generate_text_report(result)
Console().print(text_report)

# JSON export
json_report = ReportGenerator.generate_json_report(result, "momentum")
with open("report.json", "w") as f:
    json.dump(json_report, f, indent=2)
```

### ChartFactory

```python
class ChartFactory:
    """Factory for Plotly chart figures."""

    @staticmethod
    def create_equity_curve(
        result: BacktestResult,
        dark_theme: bool = False,
    ) -> go.Figure:
        """Create equity curve chart."""

    @staticmethod
    def create_drawdown_chart(
        result: BacktestResult,
        dark_theme: bool = False,
    ) -> go.Figure:
        """Create drawdown area chart."""

    @staticmethod
    def create_monthly_heatmap(
        result: BacktestResult,
        dark_theme: bool = False,
    ) -> go.Figure:
        """Create monthly returns heatmap."""

    @staticmethod
    def create_trades_chart(
        result: BacktestResult,
        close_prices: NDArray[np.float64],
        dark_theme: bool = False,
    ) -> go.Figure:
        """Create price chart with trade markers."""
```

### ResultsExporter

```python
class ResultsExporter:
    """Export backtest results to various formats."""

    @staticmethod
    def export_charts_png(
        result: BacktestResult,
        output_dir: str | Path,
        close_prices: NDArray[np.float64] | None = None,
        width: int = 1920,
        height: int = 1080,
        dark_theme: bool = False,
    ) -> None:
        """Export charts as PNG files."""

    @staticmethod
    def export_charts_html(
        result: BacktestResult,
        output_dir: str | Path,
        close_prices: NDArray[np.float64] | None = None,
        dark_theme: bool = False,
    ) -> None:
        """Export charts as interactive HTML files."""

    @staticmethod
    def export_csv(
        result: BacktestResult,
        output_dir: str | Path,
    ) -> None:
        """Export data to CSV files."""

    @staticmethod
    def export_all(
        result: BacktestResult,
        output_dir: str | Path,
        strategy_name: str = "unknown",
        close_prices: NDArray[np.float64] | None = None,
        dark_theme: bool = False,
        png_width: int = 1920,
        png_height: int = 1080,
    ) -> None:
        """Export everything to organized directory structure."""
```

**Example:**

```python
from simple_futures_backtester.output import ResultsExporter

ResultsExporter.export_all(
    result,
    output_dir="output/my_backtest",
    strategy_name="momentum",
    close_prices=close_arr,
    dark_theme=True,
)
```

**Output Structure:**

```
output/my_backtest/
├── charts/
│   ├── equity.png
│   ├── equity.html
│   ├── drawdown.png
│   ├── drawdown.html
│   ├── monthly.png
│   ├── monthly.html
│   ├── trades.png      # If close_prices provided
│   └── trades.html     # If close_prices provided
├── data/
│   ├── equity_curve.csv
│   ├── trades.csv
│   ├── metrics.csv
│   └── monthly_returns.csv
└── report.json
```

---

## Extensions API

**Module:** `simple_futures_backtester.extensions`

### FuturesPortfolio

```python
class FuturesPortfolio:
    """Wrapper for VectorBT Portfolio that applies futures point value.

    Applies point_value to PnL metrics at extraction time.
    The wrapped portfolio should be run on raw prices.

    Args:
        portfolio: VectorBT Portfolio object (already executed).
        point_value: Dollar value per point (e.g., ES = 50, NQ = 20).
        tick_size: Minimum price increment (e.g., ES = 0.25).

    Raises:
        ValueError: If point_value or tick_size is not positive.
    """

    def __init__(
        self,
        portfolio: vbt.Portfolio,
        point_value: float,
        tick_size: float,
    ) -> None: ...

    def get_analytics(self) -> PortfolioAnalytics:
        """Get dollar-denominated analytics.

        Returns:
            PortfolioAnalytics dataclass with all metrics.
        """
```

**Example:**

```python
import vectorbt as vbt
from simple_futures_backtester.extensions import FuturesPortfolio

# Create VectorBT portfolio from signals
pf = vbt.Portfolio.from_signals(
    close=es_prices,
    entries=entry_signals,
    exits=exit_signals,
    init_cash=100000.0,
)

# Wrap with ES futures specs ($50 per point, 0.25 tick size)
futures_pf = FuturesPortfolio(
    portfolio=pf,
    point_value=50.0,
    tick_size=0.25,
)

# Get dollar-denominated metrics
analytics = futures_pf.get_analytics()
print(f"Total PnL: ${analytics.total_pnl:,.2f}")
print(f"Sharpe Ratio: {analytics.sharpe_ratio:.2f}")
```

### PortfolioAnalytics

```python
@dataclass
class PortfolioAnalytics:
    """Portfolio analytics with futures-specific dollar-denominated metrics.

    Dollar-denominated metrics:
        total_pnl: Total profit/loss in dollars.
        avg_trade_pnl: Average trade profit/loss in dollars.
        avg_win_dollars: Average winning trade in dollars.
        avg_loss_dollars: Average losing trade in dollars.
        max_win_dollars: Largest winning trade in dollars.
        max_loss_dollars: Largest losing trade in dollars.
        max_drawdown_dollars: Maximum drawdown in dollars.
        total_fees: Total fees paid in dollars.
        expectancy: Expected value per trade in dollars.

    Ratio metrics (dimensionless):
        sharpe_ratio: Sharpe ratio.
        sortino_ratio: Sortino ratio.
        calmar_ratio: Calmar ratio.
        profit_factor: Gross profits / gross losses.

    Percentage metrics (decimals):
        total_return: Total return as decimal (0.15 = 15%).
        max_drawdown_percent: Maximum drawdown as decimal.
        win_rate: Winning trade percentage.

    Count metrics:
        n_trades: Total number of trades.
        n_wins: Number of winning trades.
        n_losses: Number of losing trades.

    Metadata:
        point_value: Futures point value used.
        tick_size: Minimum price increment.
    """
```

### Trailing Stop Functions

```python
def delayed_trailing_stop_nb(
    close: NDArray[np.float64],
    entry_price: float,
    entry_idx: int,
    trail_percent: float,
    activation_percent: float,
    direction: int,
) -> tuple[int, float, float]:
    """JIT-compiled delayed trailing stop.

    Two-phase trailing stop:
    1. Activation Phase: Wait for price to move favorably by activation_percent.
    2. Trailing Phase: Once activated, trail peak and exit on retracement.

    Args:
        close: Close prices array.
        entry_price: Entry price for the position.
        entry_idx: Bar index where entry occurred (0-based).
        trail_percent: Trail distance as decimal (0.02 = 2%).
        activation_percent: Required favorable move before trail activates.
        direction: Position direction (1 = long, -1 = short).

    Returns:
        Tuple of (exit_idx, exit_price, peak_price).
        exit_idx is -1 if no exit occurred.
    """
```

**Example:**

```python
import numpy as np
from simple_futures_backtester.extensions.trailing_stops import delayed_trailing_stop_nb

close = np.array([100.0, 101.0, 103.0, 102.0, 100.5, 99.0])

# Long position: activate at +1%, trail at 2%
exit_idx, exit_price, peak = delayed_trailing_stop_nb(
    close, entry_price=100.0, entry_idx=0,
    trail_percent=0.02, activation_percent=0.01, direction=1
)
print(f"Exit at bar {exit_idx} at price {exit_price}")
```

```python
def atr_trailing_stop_nb(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr: NDArray[np.float64],
    entry_idx: int,
    atr_mult: float,
    direction: int,
) -> tuple[int, float]:
    """JIT-compiled ATR-based trailing stop.

    Trailing stop with dynamic distance based on ATR.
    Activates immediately from entry (no activation threshold).

    Args:
        high: High prices array.
        low: Low prices array.
        close: Close prices array.
        atr: Pre-computed ATR array.
        entry_idx: Bar index where entry occurred.
        atr_mult: ATR multiplier for stop distance.
        direction: Position direction (1 = long, -1 = short).

    Returns:
        Tuple of (exit_idx, exit_price).
        exit_idx is -1 if no exit occurred.
    """
```

```python
def generate_trailing_exits(
    entries: NDArray[np.bool_],
    close: NDArray[np.float64],
    entry_prices: NDArray[np.float64],
    trail_percent: float,
    activation_percent: float = 0.0,
    direction: int = 1,
    high: NDArray[np.float64] | None = None,
    low: NDArray[np.float64] | None = None,
    atr: NDArray[np.float64] | None = None,
    atr_mult: float = 2.0,
    stop_type: Literal["delayed", "atr"] = "delayed",
) -> NDArray[np.bool_]:
    """Generate exit signal array from entry signals using trailing stops.

    High-level wrapper for VectorBT integration.

    Args:
        entries: Boolean array of entry signals.
        close: Close price array.
        entry_prices: Price at each entry.
        trail_percent: Trail distance as decimal.
        activation_percent: Required favorable move (delayed stop only).
        direction: 1 for long, -1 for short.
        high: High prices (required for ATR stop).
        low: Low prices (required for ATR stop).
        atr: Pre-computed ATR (required for ATR stop).
        atr_mult: ATR multiplier (ATR stop only).
        stop_type: "delayed" or "atr".

    Returns:
        Boolean array of exit signals.

    Raises:
        ValueError: If stop_type="atr" but high/low/atr not provided.
    """
```

**Example:**

```python
import numpy as np
from simple_futures_backtester.extensions.trailing_stops import generate_trailing_exits

close = np.array([100.0, 102.0, 105.0, 103.0, 100.0])
entries = np.array([True, False, False, False, False])
entry_prices = np.array([100.0, 0.0, 0.0, 0.0, 0.0])

# Generate exits for long entries with 2% delayed trailing stop
exits = generate_trailing_exits(
    entries, close, entry_prices,
    trail_percent=0.02, activation_percent=0.01, direction=1
)

# Use with VectorBT
# pf = vbt.Portfolio.from_signals(close, entries=entries, exits=exits)
```

---

## Index

### Classes

| Class | Module | Description |
|-------|--------|-------------|
| `BacktestConfig` | config | Backtest execution configuration |
| `BacktestEngine` | backtest | VectorBT portfolio wrapper |
| `BacktestResult` | backtest | Backtest results container |
| `BarSeries` | bars | Generated bar data container |
| `BaseStrategy` | strategy | Abstract strategy base class |
| `ChartFactory` | output | Plotly chart factory |
| `DataLoadError` | data | Data loading exception |
| `FuturesPortfolio` | extensions | Futures point value wrapper |
| `LoadedConfig` | config | Container for loaded config with hash |
| `ParameterSweep` | backtest | Grid search optimizer |
| `PortfolioAnalytics` | extensions | Futures analytics dataclass |
| `ReportGenerator` | output | Report formatting |
| `ResultsExporter` | output | Multi-format export |
| `Signal` | strategy | Individual signal specification |
| `StrategyConfig` | config | Strategy configuration |
| `SweepConfig` | config | Sweep configuration |
| `SweepResult` | backtest | Sweep results container |

### Functions

| Function | Module | Description |
|----------|--------|-------------|
| `atr_trailing_stop_nb` | extensions | ATR trailing stop (JIT) |
| `clear_bar_registry` | bars | Clear bar type registry |
| `clear_strategy_registry` | strategy | Clear strategy registry |
| `compute_config_hash` | config | SHA256 hash of config |
| `delayed_trailing_stop_nb` | extensions | Delayed trailing stop (JIT) |
| `generate_dollar_bars_series` | bars | Dollar bar generator |
| `generate_range_bars_series` | bars | Range bar generator |
| `generate_renko_bars_series` | bars | Renko bar generator |
| `generate_tick_bars_series` | bars | Tick bar generator |
| `generate_tick_imbalance_bars_series` | bars | Tick imbalance generator |
| `generate_trailing_exits` | extensions | Trailing exit generator |
| `generate_volume_bars_series` | bars | Volume bar generator |
| `generate_volume_imbalance_bars_series` | bars | Volume imbalance generator |
| `get_bar_generator` | bars | Get bar generator from registry |
| `get_strategy` | strategy | Get strategy from registry |
| `list_bar_types` | bars | List registered bar types |
| `list_strategies` | strategy | List registered strategies |
| `load_config` | config | Load YAML configuration |
| `load_csv` | data | Load CSV with validation |
| `load_parquet` | data | Load Parquet with validation |
| `load_strategy_config` | config | Load strategy config convenience |
| `load_sweep_config` | config | Load sweep config convenience |
| `register_bar_type` | bars | Register bar generator |
| `register_strategy` | strategy | Register strategy class |
| `unregister_bar_type` | bars | Unregister bar generator |
| `unregister_strategy` | strategy | Unregister strategy class |
