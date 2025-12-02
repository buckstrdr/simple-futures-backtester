"""Order Flow Imbalance strategy - trades exhaustion reversals.

This strategy is designed specifically for imbalance bars (volume_imbalance or
tick_imbalance), exploiting the mean-reverting nature of order flow imbalances.

Core Logic:
    When an imbalance bar closes, it signals that buying or selling pressure has
    reached an extreme (threshold). The strategy trades the reversal:

    - Imbalance bar closes BULLISH (buying exhaustion) → SHORT
    - Imbalance bar closes BEARISH (selling exhaustion) → LONG

    This is counter-trend by nature, betting that extreme imbalances
    mean-revert as the pressure that created them exhausts.

Signal Logic:
    Long (1):  Previous bar closed DOWN (selling exhaustion detected)
    Short (-1): Previous bar closed UP (buying exhaustion detected)
    Flat (0):  No signal or exit condition met

Parameters:
    min_bar_range: Minimum bar high-low range to trade (filters noise)
    exit_bars: Number of bars to hold position before exit

Example YAML config:
    strategy:
      name: order_flow_imbalance
      parameters:
        min_bar_range: 0.0005  # 0.05% minimum range
        exit_bars: 2           # Exit after 2 bars
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class OrderFlowImbalanceStrategy(BaseStrategy):
    """Order flow imbalance exhaustion reversal strategy.

    Designed for imbalance bars. Enters when imbalance threshold is hit,
    betting on mean reversion after buying/selling exhaustion.

    Key insight: When a volume imbalance bar closes, it means cumulative
    buying or selling pressure hit the threshold. This is often a local
    extreme, presenting a counter-trend opportunity.

    Attributes:
        min_bar_range: Minimum bar range to filter low-volatility bars
        exit_bars: Bars to hold position before exit
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize OrderFlowImbalanceStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
                Expected parameters:
                    - min_bar_range: Minimum bar range (default: 0.0005)
                    - exit_bars: Exit after N bars (default: 2)
        """
        super().__init__(config)
        self.min_bar_range: float = self.config.parameters.get("min_bar_range", 0.0005)
        self.exit_bars: int = self.config.parameters.get("exit_bars", 2)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate bar direction and range indicators.

        For imbalance bars, direction indicates the type of exhaustion:
        - Bullish close (close > open) = Buying exhaustion → SHORT
        - Bearish close (close < open) = Selling exhaustion → LONG

        Args:
            open_arr: Opening prices
            high_arr: High prices
            low_arr: Low prices
            close_arr: Closing prices
            volume_arr: Volume (not used)

        Returns:
            Dictionary with:
                - bar_direction: +1 bullish, -1 bearish, 0 unchanged
                - bar_range: High-low normalized by close
        """
        n = len(close_arr)

        # Bar direction: +1 bullish, -1 bearish, 0 unchanged
        bar_direction = np.zeros(n, dtype=np.float64)
        bar_direction[close_arr > open_arr] = 1.0
        bar_direction[close_arr < open_arr] = -1.0

        # Bar range (high-low) normalized by close price
        bar_range = (high_arr - low_arr) / close_arr

        return {
            "bar_direction": bar_direction,
            "bar_range": bar_range,
        }

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate order flow imbalance signals.

        Entry Logic:
            - Previous bar closed DOWN (selling exhaustion) → LONG
            - Previous bar closed UP (buying exhaustion) → SHORT
            - Bar range must exceed min_bar_range threshold

        Exit Logic:
            - Hold for exit_bars, then flatten

        Args:
            open_arr: Opening prices
            high_arr: High prices
            low_arr: Low prices
            close_arr: Closing prices
            volume_arr: Volume (optional, not used)

        Returns:
            Signal array: 1 (long), -1 (short), 0 (flat)
        """
        n = len(close_arr)

        # Handle volume array
        if volume_arr is None:
            volume_arr = np.ones(n, dtype=np.int64)

        # Calculate indicators
        indicators = self.calculate_indicators(
            open_arr, high_arr, low_arr, close_arr, volume_arr
        )

        bar_direction = indicators["bar_direction"]
        bar_range = indicators["bar_range"]

        # Initialize signals
        signals = np.zeros(n, dtype=np.int64)

        # Track position and bars held
        current_position = 0
        bars_in_position = 0

        for i in range(1, n):
            # Check if we should exit current position
            if current_position != 0:
                bars_in_position += 1

                if bars_in_position >= self.exit_bars:
                    # Exit position
                    current_position = 0
                    bars_in_position = 0
                    signals[i] = 0
                else:
                    # Hold position
                    signals[i] = current_position
                continue

            # Check entry conditions (only when flat)
            prev_bar_direction = bar_direction[i-1]
            prev_bar_range = bar_range[i-1]

            # Filter: only trade if previous bar had sufficient range
            if prev_bar_range < self.min_bar_range:
                signals[i] = 0
                continue

            # Entry: Counter-trend to imbalance direction
            if prev_bar_direction > 0:
                # Previous bar was bullish (buying exhaustion) → SHORT
                current_position = -1
                bars_in_position = 0
                signals[i] = -1

            elif prev_bar_direction < 0:
                # Previous bar was bearish (selling exhaustion) → LONG
                current_position = 1
                bars_in_position = 0
                signals[i] = 1
            else:
                # No clear direction
                signals[i] = 0

        return signals


__all__ = ["OrderFlowImbalanceStrategy"]
