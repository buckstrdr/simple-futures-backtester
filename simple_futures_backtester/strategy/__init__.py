"""Strategy framework for Simple Futures Backtester.

Provides BaseStrategy abstract class and strategy registry for
implementing custom trading strategies. Strategies generate signed
integer signals (-1, 0, 1) from OHLCV data.

Usage:
    >>> from simple_futures_backtester.strategy import BaseStrategy, Signal
    >>> from simple_futures_backtester.strategy import register_strategy, get_strategy
    >>> from simple_futures_backtester.config import StrategyConfig
    >>>
    >>> class MyStrategy(BaseStrategy):
    ...     def generate_signals(self, open_arr, high_arr, low_arr, close_arr, volume_arr):
    ...         return np.zeros(len(close_arr), dtype=np.int32)
    >>>
    >>> register_strategy("my_strategy", MyStrategy)
    >>> StrategyClass = get_strategy("my_strategy")
"""

from simple_futures_backtester.strategy.base import (
    BaseStrategy,
    Signal,
    clear_strategy_registry,
    get_strategy,
    list_strategies,
    register_strategy,
    unregister_strategy,
)

__all__: list[str] = [
    "BaseStrategy",
    "Signal",
    "clear_strategy_registry",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "unregister_strategy",
]
