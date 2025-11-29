"""Strategy framework for Simple Futures Backtester.

Provides the BaseStrategy abstract class for implementing trading strategies
that generate signals from OHLCV data. Strategies produce signed integer
signals (-1, 0, 1) representing trading direction.

Signal Values:
    -1: Short/Sell signal - Enter or maintain short position
     0: Flat/No position - Exit all positions or stay flat
     1: Long/Buy signal - Enter or maintain long position

The strategy registry allows dynamic lookup of strategy implementations by name,
enabling configuration-driven strategy selection without hardcoded dependencies.

Usage:
    >>> from simple_futures_backtester.strategy.base import BaseStrategy, Signal
    >>> from simple_futures_backtester.config import StrategyConfig
    >>> import numpy as np
    >>>
    >>> class MyStrategy(BaseStrategy):
    ...     def generate_signals(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
    ...         # Simple example: go long when close > open
    ...         signals = np.where(close_arr > open_arr, 1, 0)
    ...         return signals.astype(np.int32)
    ...
    >>> config = StrategyConfig(name="my_strategy", parameters={"threshold": 0.5})
    >>> strategy = MyStrategy(config)
    >>> signals = strategy.generate_signals(open_arr, high_arr, low_arr, close_arr, volume_arr)

Registry Usage:
    >>> from simple_futures_backtester.strategy.base import register_strategy, get_strategy
    >>> register_strategy("my_strategy", MyStrategy)
    >>> StrategyClass = get_strategy("my_strategy")
    >>> strategy = StrategyClass(config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig


@dataclass
class Signal:
    """Individual trading signal specification.

    Used for representing a single signal with direction and optional size
    modifier. While generate_signals() returns arrays for vectorized operations,
    this dataclass provides a structured representation for individual signals.

    Attributes:
        direction: Signal direction (-1 for short, 0 for flat, 1 for long).
        size: Position size multiplier (default=1.0).
            Size of 1.0 means standard position sizing from BacktestConfig.
            Size of 2.0 means double the standard size.
            Size of 0.5 means half the standard size.

    Example:
        >>> signal = Signal(direction=1, size=1.0)  # Standard long
        >>> signal = Signal(direction=-1, size=0.5)  # Half-size short
        >>> signal = Signal(direction=0)  # Flat, size doesn't matter
    """

    direction: int
    size: float = 1.0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Strategies generate signed integer signals (-1, 0, 1) from OHLCV data.
    Signal values indicate trading direction:
        -1: Short/Sell signal
         0: Flat/No position
         1: Long/Buy signal

    Subclasses must implement generate_signals() to define their trading logic.
    The optional calculate_indicators() method can be overridden to compute
    technical indicators used in signal generation.

    Attributes:
        config: StrategyConfig containing strategy name and parameters.

    Example:
        >>> class MomentumStrategy(BaseStrategy):
        ...     def __init__(self, config: StrategyConfig):
        ...         super().__init__(config)
        ...         self.rsi_period = self.config.parameters.get('rsi_period', 14)
        ...
        ...     def generate_signals(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
        ...         signals = np.zeros(len(close_arr), dtype=np.int32)
        ...         # Add strategy logic here
        ...         return signals
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize strategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
                Parameters can be accessed via self.config.parameters dict.
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

        This is the core method that defines strategy behavior. Implementations
        should analyze the price and volume data to produce directional signals.

        Args:
            open_arr: Opening prices as float64 array.
            high_arr: High prices as float64 array.
            low_arr: Low prices as float64 array.
            close_arr: Closing prices as float64 array.
            volume_arr: Volume as int64 array.

        Returns:
            Signal array as int32 with values -1 (short), 0 (flat), 1 (long).
            Array length MUST match close_arr length.

        Note:
            - Initial warmup bars (before indicators are ready) should be 0
            - Signals are applied to the bar at the same index
            - Only values -1, 0, and 1 are valid
        """
        pass

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],  # noqa: ARG002
        high_arr: NDArray[np.float64],  # noqa: ARG002
        low_arr: NDArray[np.float64],  # noqa: ARG002
        close_arr: NDArray[np.float64],  # noqa: ARG002
        volume_arr: NDArray[np.int64],  # noqa: ARG002
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate technical indicators for this strategy.

        This is a helper method that subclasses can override to compute
        indicators used in signal generation. Separating indicator calculation
        from signal generation enables indicator reuse and easier debugging.

        The default implementation returns an empty dictionary. Subclasses
        should override this method to compute their required indicators.

        Args:
            open_arr: Opening prices as float64 array.
            high_arr: High prices as float64 array.
            low_arr: Low prices as float64 array.
            close_arr: Closing prices as float64 array.
            volume_arr: Volume as int64 array.

        Returns:
            Dictionary mapping indicator names to float64 arrays.
            All arrays should have the same length as close_arr.

        Example:
            >>> def calculate_indicators(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
            ...     return {
            ...         "rsi": compute_rsi(close_arr, self.rsi_period),
            ...         "ema_fast": compute_ema(close_arr, self.fast_period),
            ...         "ema_slow": compute_ema(close_arr, self.slow_period),
            ...     }
        """
        return {}


# Strategy Registry
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str, strategy_class: type[BaseStrategy]) -> None:
    """Register a strategy class with the factory.

    Strategy classes are registered globally and can be retrieved by name using
    get_strategy(). This enables dynamic strategy selection based on configuration
    without hardcoding strategy dependencies.

    Args:
        name: Strategy identifier (e.g., "momentum", "mean_reversion").
            Should be lowercase and use underscores for multi-word names.
        strategy_class: Strategy class extending BaseStrategy.
            Must be a class (not an instance) that inherits from BaseStrategy.

    Raises:
        ValueError: If name is empty or strategy_class is not a BaseStrategy subclass.

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def generate_signals(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
        ...         return np.zeros(len(close_arr), dtype=np.int32)
        >>> register_strategy("my_strategy", MyStrategy)

    Note:
        Registering with an existing name will overwrite the previous
        strategy without warning. This allows for testing and hot-reloading.
    """
    if not name:
        raise ValueError("Strategy name cannot be empty")
    if not isinstance(strategy_class, type) or not issubclass(strategy_class, BaseStrategy):
        raise ValueError(
            f"Strategy must be a BaseStrategy subclass, got {type(strategy_class).__name__}"
        )

    _STRATEGY_REGISTRY[name] = strategy_class


def get_strategy(name: str) -> type[BaseStrategy]:
    """Retrieve a registered strategy class by name.

    Args:
        name: Strategy identifier (e.g., "momentum", "mean_reversion").

    Returns:
        The registered strategy class (NOT an instance).
        Use the returned class to create strategy instances with a config.

    Raises:
        KeyError: If strategy is not registered. Error message includes
            list of available strategies for easier debugging.

    Example:
        >>> StrategyClass = get_strategy("momentum")
        >>> config = StrategyConfig(name="momentum", parameters={"rsi_period": 14})
        >>> strategy = StrategyClass(config)
        >>> signals = strategy.generate_signals(open_arr, high_arr, low_arr, close_arr, volume_arr)
    """
    if name not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        if available:
            raise KeyError(f"Unknown strategy '{name}'. Available strategies: {available}")
        raise KeyError(f"Unknown strategy '{name}'. No strategies registered.")
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """List all registered strategies.

    Returns:
        List of strategy identifiers in alphabetical order.

    Example:
        >>> list_strategies()
        ['breakout', 'mean_reversion', 'momentum']
    """
    return sorted(_STRATEGY_REGISTRY.keys())


def unregister_strategy(name: str) -> bool:
    """Unregister a strategy by name.

    Primarily useful for testing to clean up registered strategies.

    Args:
        name: Strategy identifier to remove.

    Returns:
        True if the strategy was removed, False if it wasn't registered.
    """
    if name in _STRATEGY_REGISTRY:
        del _STRATEGY_REGISTRY[name]
        return True
    return False


def clear_strategy_registry() -> None:
    """Clear all registered strategies.

    Primarily useful for testing to reset the registry state.
    """
    _STRATEGY_REGISTRY.clear()


__all__: list[str] = [
    "BaseStrategy",
    "Signal",
    "clear_strategy_registry",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "unregister_strategy",
]
