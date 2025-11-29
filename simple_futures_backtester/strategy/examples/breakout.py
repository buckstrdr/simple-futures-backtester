"""Breakout trading strategy implementation.

This strategy uses EMA crossovers to identify trend breakouts. Long signals
are generated when the fast EMA crosses above the slow EMA. Short signals
are generated when the fast EMA crosses below the slow EMA.

Signal Logic:
    Long (1):  fast_ema crosses above slow_ema
    Short (-1): fast_ema crosses below slow_ema
    Flat (0):  no crossover event

Note: This generates EVENT signals (signal only on crossover bar),
not STATE signals (signal persists while condition holds).

Parameters (from StrategyConfig.parameters):
    fast_period: Fast EMA period (default: 10)
    slow_period: Slow EMA period (default: 30)

Example YAML config:
    strategy:
      name: breakout
      parameters:
        fast_period: 10
        slow_period: 30
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """EMA crossover breakout trading strategy.

    Generates long signals when fast EMA crosses above slow EMA.
    Generates short signals when fast EMA crosses below slow EMA.
    Otherwise returns flat (0).

    The strategy uses VectorBT's MA indicator (with ewm=True for EMA)
    and the crossed_above/crossed_below signal helpers. During the
    indicator warmup period (first slow_period bars), signals will
    be flat due to NaN comparisons returning False.

    Attributes:
        fast_period: Fast EMA calculation period.
        slow_period: Slow EMA calculation period.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize BreakoutStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
                Expected parameters:
                    - fast_period: Fast EMA period (default: 10)
                    - slow_period: Slow EMA period (default: 30)
        """
        super().__init__(config)
        self.fast_period: int = self.config.parameters.get("fast_period", 10)
        self.slow_period: int = self.config.parameters.get("slow_period", 30)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate EMA indicators.

        Uses VectorBT's MA indicator with ewm=True for exponential
        moving average calculation.

        Args:
            open_arr: Opening prices (not used for this strategy).
            high_arr: High prices (not used for this strategy).
            low_arr: Low prices (not used for this strategy).
            close_arr: Closing prices used for EMA calculation.
            volume_arr: Volume (not used for this strategy).

        Returns:
            Dictionary containing:
                - 'fast_ema': Fast EMA values as float64 array
                - 'slow_ema': Slow EMA values as float64 array
        """
        # Convert close prices to pandas Series for VectorBT
        close_series = pd.Series(close_arr)

        # Calculate fast EMA (ewm=True for exponential moving average)
        fast_ema_indicator = vbt.MA.run(
            close_series, window=self.fast_period, ewm=True
        )
        fast_ema = fast_ema_indicator.ma.values

        # Calculate slow EMA (ewm=True for exponential moving average)
        slow_ema_indicator = vbt.MA.run(
            close_series, window=self.slow_period, ewm=True
        )
        slow_ema = slow_ema_indicator.ma.values

        return {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
        }

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        """Generate breakout signals from OHLCV data.

        Long when: fast_ema crosses above slow_ema
        Short when: fast_ema crosses below slow_ema
        Otherwise: Flat (0)

        Detects crossover EVENTS by comparing current and previous states.
        Signals are only generated on the bar where the crossover occurs.

        During the warmup period (while indicators contain NaN values),
        signals will be flat because NaN comparisons return False.

        Args:
            open_arr: Opening prices (passed to calculate_indicators).
            high_arr: High prices (passed to calculate_indicators).
            low_arr: Low prices (passed to calculate_indicators).
            close_arr: Closing prices for indicator calculation.
            volume_arr: Volume (passed to calculate_indicators).

        Returns:
            Signal array with values -1 (short), 0 (flat), 1 (long).
            Array length matches close_arr length.
        """
        # Initialize all signals to flat (0)
        signals = np.zeros(len(close_arr), dtype=np.int32)

        # Convert close to pandas Series for VectorBT
        close_series = pd.Series(close_arr)

        # Calculate EMAs (returns pandas Series)
        fast_ema = vbt.MA.run(close_series, window=self.fast_period, ewm=True).ma
        slow_ema = vbt.MA.run(close_series, window=self.slow_period, ewm=True).ma

        # Detect crossover events manually
        # Current state: is fast above slow?
        fast_above_slow = fast_ema > slow_ema
        # Previous state: was fast above slow?
        # Fill NaN with False to handle first bar (no previous state)
        fast_above_slow_prev = fast_above_slow.shift(1)
        fast_above_slow_prev = fast_above_slow_prev.fillna(False).astype(bool)

        # Crossed above: wasn't above before, is above now
        crossed_above = (fast_above_slow_prev == False) & fast_above_slow  # noqa: E712
        # Crossed below: was above before, isn't above now
        crossed_below = fast_above_slow_prev & (fast_above_slow == False)  # noqa: E712

        # Convert to numpy arrays and assign signals
        long_entry = crossed_above.values
        short_entry = crossed_below.values

        signals[long_entry] = 1
        signals[short_entry] = -1

        return signals
