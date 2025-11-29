"""Mean reversion trading strategy implementation.

This strategy uses Bollinger Bands to identify oversold and overbought
conditions. Long signals are generated when price touches or falls below
the lower band. Short signals are generated when price touches or rises
above the upper band.

Signal Logic:
    Long (1):  close <= lower_band (oversold condition)
    Short (-1): close >= upper_band (overbought condition)
    Flat (0):  price between bands

Note: This generates STATE signals (signal persists while condition holds),
not EVENT signals (signal only on single bar).

Parameters (from StrategyConfig.parameters):
    bb_period: Bollinger Bands lookback period (default: 20)
    bb_std: Number of standard deviations for bands (default: 2.0)

Example YAML config:
    strategy:
      name: mean_reversion
      parameters:
        bb_period: 20
        bb_std: 2.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion trading strategy.

    Generates long signals when price touches or falls below the lower
    Bollinger Band (oversold). Generates short signals when price touches
    or rises above the upper Bollinger Band (overbought).

    The strategy uses VectorBT's BBANDS indicator for calculation.
    During the indicator warmup period (first bb_period bars), signals
    will be flat due to NaN comparisons returning False.

    Attributes:
        bb_period: Bollinger Bands lookback period.
        bb_std: Number of standard deviations for band width.
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize MeanReversionStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
                Expected parameters:
                    - bb_period: Bollinger Bands lookback period (default: 20)
                    - bb_std: Number of standard deviations (default: 2.0)
        """
        super().__init__(config)
        self.bb_period: int = self.config.parameters.get("bb_period", 20)
        self.bb_std: float = self.config.parameters.get("bb_std", 2.0)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate Bollinger Bands indicators.

        Uses VectorBT's BBANDS indicator for calculation. The alpha
        parameter controls the number of standard deviations for band width.

        Args:
            open_arr: Opening prices (not used for this strategy).
            high_arr: High prices (not used for this strategy).
            low_arr: Low prices (not used for this strategy).
            close_arr: Closing prices used for Bollinger Bands calculation.
            volume_arr: Volume (not used for this strategy).

        Returns:
            Dictionary containing:
                - 'upper_band': Upper Bollinger Band values
                - 'middle_band': Middle band (SMA) values
                - 'lower_band': Lower Bollinger Band values
        """
        # Convert close prices to pandas Series for VectorBT
        close_series = pd.Series(close_arr)

        # Calculate Bollinger Bands using VectorBT
        bb_indicator = vbt.BBANDS.run(
            close_series, window=self.bb_period, alpha=self.bb_std
        )

        return {
            "upper_band": bb_indicator.upper.values,
            "middle_band": bb_indicator.middle.values,
            "lower_band": bb_indicator.lower.values,
        }

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        """Generate mean reversion signals from OHLCV data.

        Long when: close <= lower_band (oversold)
        Short when: close >= upper_band (overbought)
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
        lower_band = indicators["lower_band"]
        upper_band = indicators["upper_band"]

        # Generate long signals (price <= lower band - oversold)
        long_condition = close_arr <= lower_band
        signals[long_condition] = 1

        # Generate short signals (price >= upper band - overbought)
        short_condition = close_arr >= upper_band
        signals[short_condition] = -1

        return signals
