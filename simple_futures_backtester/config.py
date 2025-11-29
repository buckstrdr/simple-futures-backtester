"""Configuration system for Simple Futures Backtester.

Provides dataclass definitions for strategy, backtest, and sweep configurations
with YAML loading support, environment variable overrides, and SHA256 hash
generation for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy.

    Attributes:
        name: Strategy identifier (e.g., "momentum", "mean_reversion").
        parameters: Strategy-specific parameters as key-value pairs.
    """

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for backtesting execution.

    Attributes:
        initial_capital: Starting capital for the backtest.
        fees: Transaction fees as a decimal (e.g., 0.0001 = 0.01%).
        slippage: Slippage as a decimal (e.g., 0.0001 = 0.01%).
        size: Position size (number of contracts or units).
        size_type: How size is interpreted ("fixed", "percent", "target").
        freq: Rebalancing frequency (e.g., "1D", "1H", "5m").
    """

    initial_capital: float = 100000.0
    fees: float = 0.0001
    slippage: float = 0.0001
    size: int = 1
    size_type: str = "fixed"
    freq: str = "1D"


@dataclass
class SweepConfig:
    """Configuration for parameter sweep optimization.

    Attributes:
        strategy: Name of the strategy to sweep.
        parameters: Parameter grid where keys are parameter names and
                   values are lists of values to test.
        backtest_overrides: Optional overrides for BacktestConfig fields
                           during the sweep.
    """

    strategy: str
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    backtest_overrides: dict[str, Any] = field(default_factory=dict)


def compute_config_hash(yaml_content: str) -> str:
    """Compute SHA256 hash of YAML content for reproducibility tracking.

    Args:
        yaml_content: Raw YAML file content as a string.

    Returns:
        Hexadecimal SHA256 hash of the normalized content.
    """
    normalized = yaml_content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _apply_env_overrides_backtest(config: BacktestConfig) -> BacktestConfig:
    """Apply environment variable overrides to BacktestConfig.

    Environment variables follow the pattern SFB_BACKTEST_<FIELD>.
    Supported overrides:
        - SFB_BACKTEST_INITIAL_CAPITAL or SFB_CAPITAL
        - SFB_BACKTEST_FEES or SFB_FEES
        - SFB_BACKTEST_SLIPPAGE or SFB_SLIPPAGE
        - SFB_BACKTEST_SIZE or SFB_SIZE
        - SFB_BACKTEST_SIZE_TYPE
        - SFB_BACKTEST_FREQ

    Args:
        config: The BacktestConfig to potentially override.

    Returns:
        BacktestConfig with any environment overrides applied.
    """
    initial_capital = os.environ.get("SFB_BACKTEST_INITIAL_CAPITAL", os.environ.get("SFB_CAPITAL"))
    if initial_capital is not None:
        config = BacktestConfig(
            initial_capital=float(initial_capital),
            fees=config.fees,
            slippage=config.slippage,
            size=config.size,
            size_type=config.size_type,
            freq=config.freq,
        )

    fees = os.environ.get("SFB_BACKTEST_FEES", os.environ.get("SFB_FEES"))
    if fees is not None:
        config = BacktestConfig(
            initial_capital=config.initial_capital,
            fees=float(fees),
            slippage=config.slippage,
            size=config.size,
            size_type=config.size_type,
            freq=config.freq,
        )

    slippage = os.environ.get("SFB_BACKTEST_SLIPPAGE", os.environ.get("SFB_SLIPPAGE"))
    if slippage is not None:
        config = BacktestConfig(
            initial_capital=config.initial_capital,
            fees=config.fees,
            slippage=float(slippage),
            size=config.size,
            size_type=config.size_type,
            freq=config.freq,
        )

    size = os.environ.get("SFB_BACKTEST_SIZE", os.environ.get("SFB_SIZE"))
    if size is not None:
        config = BacktestConfig(
            initial_capital=config.initial_capital,
            fees=config.fees,
            slippage=config.slippage,
            size=int(size),
            size_type=config.size_type,
            freq=config.freq,
        )

    size_type = os.environ.get("SFB_BACKTEST_SIZE_TYPE")
    if size_type is not None:
        config = BacktestConfig(
            initial_capital=config.initial_capital,
            fees=config.fees,
            slippage=config.slippage,
            size=config.size,
            size_type=size_type,
            freq=config.freq,
        )

    freq = os.environ.get("SFB_BACKTEST_FREQ")
    if freq is not None:
        config = BacktestConfig(
            initial_capital=config.initial_capital,
            fees=config.fees,
            slippage=config.slippage,
            size=config.size,
            size_type=config.size_type,
            freq=freq,
        )

    return config


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


def load_config(config_path: str | Path) -> LoadedConfig:
    """Load configuration from a YAML file with environment overrides.

    This function reads a YAML configuration file and materializes it into
    typed dataclass objects. Environment variables can override backtest
    settings using the SFB_* prefix.

    The function is deterministic and side-effect free (except for file I/O),
    making it suitable for testing with mocked file content.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        LoadedConfig containing the parsed configuration and its hash.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    yaml_content = path.read_text(encoding="utf-8")
    config_hash = compute_config_hash(yaml_content)

    data = yaml.safe_load(yaml_content)
    if data is None:
        data = {}

    strategy_config: StrategyConfig | None = None
    backtest_config = BacktestConfig()
    sweep_config: SweepConfig | None = None

    if "strategy" in data:
        strategy_data = data["strategy"]
        if isinstance(strategy_data, dict):
            if "name" not in strategy_data:
                raise ValueError("Strategy configuration requires 'name' field")
            strategy_config = StrategyConfig(
                name=strategy_data["name"],
                parameters=strategy_data.get("parameters", {}),
            )

    if "backtest" in data:
        bt_data = data["backtest"]
        if isinstance(bt_data, dict):
            backtest_config = BacktestConfig(
                initial_capital=float(bt_data.get("initial_capital", 100000.0)),
                fees=float(bt_data.get("fees", 0.0001)),
                slippage=float(bt_data.get("slippage", 0.0001)),
                size=int(bt_data.get("size", 1)),
                size_type=str(bt_data.get("size_type", "fixed")),
                freq=str(bt_data.get("freq", "1D")),
            )

    if "sweep" in data:
        sweep_data = data["sweep"]
        if isinstance(sweep_data, dict):
            if "strategy" not in sweep_data:
                raise ValueError("Sweep configuration requires 'strategy' field")
            sweep_config = SweepConfig(
                strategy=sweep_data["strategy"],
                parameters=sweep_data.get("parameters", {}),
                backtest_overrides=sweep_data.get("backtest_overrides", {}),
            )

    backtest_config = _apply_env_overrides_backtest(backtest_config)

    return LoadedConfig(
        strategy=strategy_config,
        backtest=backtest_config,
        sweep=sweep_config,
        config_hash=config_hash,
        source_path=str(path.resolve()),
    )


def load_strategy_config(config_path: str | Path) -> tuple[StrategyConfig, BacktestConfig, str]:
    """Convenience function to load a strategy configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (StrategyConfig, BacktestConfig, config_hash).

    Raises:
        ValueError: If the config file does not contain a strategy section.
    """
    loaded = load_config(config_path)
    if loaded.strategy is None:
        raise ValueError(f"Config file does not contain strategy section: {config_path}")
    return loaded.strategy, loaded.backtest, loaded.config_hash


def load_sweep_config(config_path: str | Path) -> tuple[SweepConfig, BacktestConfig, str]:
    """Convenience function to load a sweep configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (SweepConfig, BacktestConfig, config_hash).

    Raises:
        ValueError: If the config file does not contain a sweep section.
    """
    loaded = load_config(config_path)
    if loaded.sweep is None:
        raise ValueError(f"Config file does not contain sweep section: {config_path}")

    merged_backtest = loaded.backtest
    if loaded.sweep.backtest_overrides:
        overrides = loaded.sweep.backtest_overrides
        merged_backtest = BacktestConfig(
            initial_capital=float(
                overrides.get("initial_capital", merged_backtest.initial_capital)
            ),
            fees=float(overrides.get("fees", merged_backtest.fees)),
            slippage=float(overrides.get("slippage", merged_backtest.slippage)),
            size=int(overrides.get("size", merged_backtest.size)),
            size_type=str(overrides.get("size_type", merged_backtest.size_type)),
            freq=str(overrides.get("freq", merged_backtest.freq)),
        )

    return loaded.sweep, merged_backtest, loaded.config_hash
