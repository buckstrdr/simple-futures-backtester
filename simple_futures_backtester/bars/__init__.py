"""Alternative bar generation modules.

Provides JIT-compiled generators for alternative bar types:
- Renko bars
- Range bars
- Tick bars
- Volume bars
- Dollar bars
- Imbalance bars (tick and volume)

All generators target 1M+ rows/sec throughput via Numba JIT compilation.

Usage:
    >>> from simple_futures_backtester.bars import BarSeries, register_bar_type
    >>> from simple_futures_backtester.bars import get_bar_generator, list_bar_types
    >>>
    >>> # Register a custom bar generator
    >>> def my_generator(open, high, low, close, volume, **kwargs) -> BarSeries:
    ...     # Implementation
    ...     pass
    >>> register_bar_type("my_bars", my_generator)
    >>>
    >>> # Retrieve and use a bar generator
    >>> generator = get_bar_generator("my_bars")
    >>> bars = generator(open_arr, high_arr, low_arr, close_arr, volume_arr)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.utils.jit_utils import ensure_float64, ensure_int64

if TYPE_CHECKING:
    pass


class BarGeneratorProtocol(Protocol):
    """Protocol defining the signature for bar generator functions.

    All bar generators must accept OHLCV arrays and keyword arguments,
    returning a BarSeries instance.
    """

    def __call__(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
        **kwargs: Any,
    ) -> BarSeries:
        """Generate bars from OHLCV data.

        Args:
            open_arr: Opening prices as float64 array.
            high_arr: High prices as float64 array.
            low_arr: Low prices as float64 array.
            close_arr: Closing prices as float64 array.
            volume_arr: Volume as int64 array.
            **kwargs: Bar-type-specific parameters.

        Returns:
            BarSeries containing the generated bars.
        """
        ...


BarGeneratorFunc = Callable[..., "BarSeries"]


@dataclass
class BarSeries:
    """Container for alternative bar series with provenance tracking.

    BarSeries stores aggregated price bars along with metadata about how they
    were generated. The index_map array provides a mapping from each bar index
    back to the original source data row, enabling signal alignment and debugging.

    All price arrays are normalized to float64 and volume/indices to int64
    during initialization to ensure JIT-safety.

    Attributes:
        type: Bar type identifier (e.g., "renko", "range", "volume", "tick").
        parameters: Bar-specific generation parameters (e.g., {"brick_size": 10}).
            These parameters are included in hash computation for provenance.
        open: Opening prices as float64 numpy array.
        high: High prices as float64 numpy array.
        low: Low prices as float64 numpy array.
        close: Closing prices as float64 numpy array.
        volume: Aggregated volume as int64 numpy array.
        index_map: Mapping from bar index to source data row index (int64).
            For each bar, this indicates the source row where the bar completed.

    Example:
        >>> import numpy as np
        >>> bars = BarSeries(
        ...     type="renko",
        ...     parameters={"brick_size": 10},
        ...     open=np.array([100.0, 110.0]),
        ...     high=np.array([110.0, 120.0]),
        ...     low=np.array([100.0, 110.0]),
        ...     close=np.array([110.0, 120.0]),
        ...     volume=np.array([1000, 2000]),
        ...     index_map=np.array([50, 120]),
        ... )
        >>> len(bars)
        2
        >>> bars.type
        'renko'
    """

    type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    open: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    high: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    low: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    close: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    volume: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))
    index_map: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        """Validate and normalize BarSeries arrays after initialization.

        Ensures all price arrays are float64, volume and index_map are int64,
        and all arrays have consistent lengths.

        Raises:
            ValueError: If arrays have inconsistent lengths or cannot be converted.
        """
        self.open = ensure_float64(self.open)
        self.high = ensure_float64(self.high)
        self.low = ensure_float64(self.low)
        self.close = ensure_float64(self.close)
        self.volume = ensure_int64(self.volume)
        self.index_map = ensure_int64(self.index_map)

        lengths = {
            "open": len(self.open),
            "high": len(self.high),
            "low": len(self.low),
            "close": len(self.close),
            "volume": len(self.volume),
            "index_map": len(self.index_map),
        }

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"BarSeries arrays have inconsistent lengths: {lengths}")

    def __len__(self) -> int:
        """Return the number of bars in the series."""
        return len(self.close)

    @property
    def is_empty(self) -> bool:
        """Check if the bar series contains no bars."""
        return len(self.close) == 0


_BAR_REGISTRY: dict[str, BarGeneratorFunc] = {}


def register_bar_type(name: str, generator_func: BarGeneratorFunc) -> None:
    """Register a bar generator function with the factory.

    Bar generators are registered globally and can be retrieved by name using
    get_bar_generator(). This enables dynamic bar type selection based on
    configuration without hardcoding bar type dependencies.

    Args:
        name: Bar type identifier (e.g., "renko", "range", "volume").
            Should be lowercase and use underscores for multi-word names.
        generator_func: Function that generates bars from OHLCV data.
            Must accept (open, high, low, close, volume, **kwargs) and
            return a BarSeries instance.

    Raises:
        ValueError: If name is empty or generator_func is not callable.

    Example:
        >>> def generate_custom_bars(open, high, low, close, volume, **kwargs):
        ...     # Custom bar generation logic
        ...     return BarSeries(type="custom", parameters=kwargs, ...)
        >>> register_bar_type("custom", generate_custom_bars)

    Note:
        Registering with an existing name will overwrite the previous
        generator without warning. This allows for testing and hot-reloading.
    """
    if not name:
        raise ValueError("Bar type name cannot be empty")
    if not callable(generator_func):
        raise ValueError(f"Generator must be callable, got {type(generator_func).__name__}")

    _BAR_REGISTRY[name] = generator_func


def get_bar_generator(name: str) -> BarGeneratorFunc:
    """Retrieve a registered bar generator by name.

    Args:
        name: Bar type identifier (e.g., "renko", "range", "volume").

    Returns:
        The registered bar generator function.

    Raises:
        KeyError: If bar type is not registered. Error message includes
            list of available bar types for easier debugging.

    Example:
        >>> generator = get_bar_generator("renko")
        >>> bars = generator(open, high, low, close, volume, brick_size=10)
    """
    if name not in _BAR_REGISTRY:
        available = list(_BAR_REGISTRY.keys())
        if available:
            raise KeyError(f"Unknown bar type '{name}'. Available types: {available}")
        raise KeyError(f"Unknown bar type '{name}'. No bar types registered.")
    return _BAR_REGISTRY[name]


def list_bar_types() -> list[str]:
    """List all registered bar types.

    Returns:
        List of bar type identifiers in alphabetical order.

    Example:
        >>> list_bar_types()
        ['range', 'renko', 'tick', 'volume']
    """
    return sorted(_BAR_REGISTRY.keys())


def unregister_bar_type(name: str) -> bool:
    """Unregister a bar generator by name.

    Primarily useful for testing to clean up registered bar types.

    Args:
        name: Bar type identifier to remove.

    Returns:
        True if the bar type was removed, False if it wasn't registered.
    """
    if name in _BAR_REGISTRY:
        del _BAR_REGISTRY[name]
        return True
    return False


def clear_bar_registry() -> None:
    """Clear all registered bar types.

    Primarily useful for testing to reset the registry state.
    """
    _BAR_REGISTRY.clear()


__all__: list[str] = [
    "BarGeneratorFunc",
    "BarGeneratorProtocol",
    "BarSeries",
    "clear_bar_registry",
    "get_bar_generator",
    "list_bar_types",
    "register_bar_type",
    "unregister_bar_type",
]

# Auto-import bar generators to trigger registration
from simple_futures_backtester.bars import renko  # noqa: F401, E402
from simple_futures_backtester.bars import range_bars  # noqa: F401, E402
from simple_futures_backtester.bars import tick_bars  # noqa: F401, E402
from simple_futures_backtester.bars import volume_bars  # noqa: F401, E402
from simple_futures_backtester.bars import dollar_bars  # noqa: F401, E402
from simple_futures_backtester.bars import imbalance_bars  # noqa: F401, E402
