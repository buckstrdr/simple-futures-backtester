"""Momentum trading strategy implementation.

This strategy combines RSI and EMA crossovers to identify momentum-based
trading opportunities. Long signals are generated when RSI is above 50 and
the fast EMA is above the slow EMA. Short signals are generated when RSI
is below 50 and the fast EMA is below the slow EMA.

Signal Logic:
    Long (1):  RSI > 50 AND fast_ema > slow_ema
    Short (-1): RSI < 50 AND fast_ema < slow_ema
    Flat (0):  All other conditions

Indicators:
    RSI: Calculated using Wilder's smoothing method (pandas EWM with alpha=1/period)
    EMA: Calculated using pandas EWM with span parameter

Parameters (from StrategyConfig.parameters):
    rsi_period: RSI lookback period (default: 14)
    fast_ema: Fast EMA period (default: 9)
    slow_ema: Slow EMA period (default: 21)

Example YAML config:
    strategy:
      name: momentum
      parameters:
        rsi_period: 14
        fast_ema: 9
        slow_ema: 21
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """RSI + EMA momentum-based trading strategy.

    Generates long signals when RSI > 50 AND fast EMA > slow EMA.
    Generates short signals when RSI < 50 AND fast EMA < slow EMA.
    Otherwise returns flat (0).

    The strategy uses pandas for indicator calculation (Wilder's RSI and
    exponential moving averages). During the indicator warmup period
    (first N bars where N is the longest lookback), signals will be flat
    due to NaN comparisons returning False.

    Attributes:
        rsi_period: RSI calculation period.
        fast_ema: Fast EMA period for momentum detection.
        slow_ema: Slow EMA period for trend confirmation.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize MomentumStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
                Expected parameters:
                    - rsi_period: RSI lookback period (default: 14)
                    - fast_ema: Fast EMA period (default: 9)
                    - slow_ema: Slow EMA period (default: 21)
        """
        super().__init__(config)
        self.rsi_period: int = self.config.parameters.get("rsi_period", 14)
        self.fast_ema: int = self.config.parameters.get("fast_ema", 9)
        self.slow_ema: int = self.config.parameters.get("slow_ema", 21)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate RSI and EMA indicators.

        Uses pandas for RSI calculation (Wilder's smoothing method) and EMA
        calculation. The RSI is calculated using the standard formula:
        RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss.

        Args:
            open_arr: Opening prices (not used for this strategy).
            high_arr: High prices (not used for this strategy).
            low_arr: Low prices (not used for this strategy).
            close_arr: Closing prices used for RSI and EMA calculation.
            volume_arr: Volume (not used for this strategy).

        Returns:
            Dictionary containing:
                - 'rsi': RSI values as float64 array
                - 'fast_ema': Fast EMA values as float64 array
                - 'slow_ema': Slow EMA values as float64 array
        """
        # Convert close prices to pandas Series
        close_series = pd.Series(close_arr)

        # Calculate RSI using pandas
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Use Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0).values  # Fill NaN with neutral value

        # Calculate fast EMA using pandas
        fast_ema = close_series.ewm(span=self.fast_ema, adjust=False).mean().values

        # Calculate slow EMA using pandas
        slow_ema = close_series.ewm(span=self.slow_ema, adjust=False).mean().values

        return {
            "rsi": rsi,
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
        """Generate momentum signals from OHLCV data.

        Long when: RSI > 50 AND fast_ema > slow_ema
        Short when: RSI < 50 AND fast_ema < slow_ema
        Otherwise: Flat (0)

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

        # Calculate indicators
        indicators = self.calculate_indicators(
            open_arr, high_arr, low_arr, close_arr, volume_arr
        )
        rsi = indicators["rsi"]
        fast_ema = indicators["fast_ema"]
        slow_ema = indicators["slow_ema"]

        # Generate long signals (RSI > 50 AND fast > slow)
        long_condition = (rsi > 50) & (fast_ema > slow_ema)
        signals[long_condition] = 1

        # Generate short signals (RSI < 50 AND fast < slow)
        short_condition = (rsi < 50) & (fast_ema < slow_ema)
        signals[short_condition] = -1

        return signals
