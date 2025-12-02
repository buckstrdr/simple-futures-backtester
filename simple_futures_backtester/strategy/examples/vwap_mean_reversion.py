"""VWAP Mean Reversion Strategy.

Tests multiple VWAP anchoring methods:
  1. Session VWAP (resets each trading day)
  2. Weekly VWAP (resets each week)
  3. Anchored VWAP (from specific date/event)

Core Logic:
  - Price deviation from VWAP indicates temporary imbalance
  - Large deviations tend to revert (institutional support/resistance)
  - Volume-weighted price = fair value proxy
  - Entry on deviation, exit on reversion to VWAP

Different from technical indicators:
  - VWAP is volume-weighted (institutional footprint)
  - Acts as dynamic support/resistance
  - Self-fulfilling (algos use it for execution)
"""
import numpy as np
from numpy.typing import NDArray
from simple_futures_backtester.strategy.base import BaseStrategy


class VWAPMeanReversionStrategy(BaseStrategy):
    """Mean reversion to VWAP with multiple anchoring options."""

    def _calculate_vwap(
        self,
        high: NDArray,
        low: NDArray,
        close: NDArray,
        volume: NDArray,
        anchor_points: NDArray,
    ) -> NDArray:
        """Calculate VWAP with anchoring support.

        Args:
            high, low, close: Price arrays
            volume: Volume array
            anchor_points: Boolean array where True = reset VWAP

        Returns:
            VWAP array
        """
        n = len(close)
        vwap = np.zeros(n, dtype=np.float64)

        # Typical price
        typical_price = (high + low + close) / 3

        # Initialize accumulators
        cum_vol = 0.0
        cum_tp_vol = 0.0

        for i in range(n):
            # Reset on anchor points
            if anchor_points[i]:
                cum_vol = 0.0
                cum_tp_vol = 0.0

            # Accumulate
            cum_vol += volume[i]
            cum_tp_vol += typical_price[i] * volume[i]

            # Calculate VWAP
            if cum_vol > 0:
                vwap[i] = cum_tp_vol / cum_vol
            else:
                vwap[i] = close[i]

        return vwap

    def _calculate_std_dev(
        self,
        close: NDArray,
        vwap: NDArray,
        volume: NDArray,
        anchor_points: NDArray,
    ) -> NDArray:
        """Calculate volume-weighted standard deviation from VWAP."""
        n = len(close)
        std_dev = np.zeros(n, dtype=np.float64)

        cum_vol = 0.0
        cum_sq_diff = 0.0

        for i in range(n):
            if anchor_points[i]:
                cum_vol = 0.0
                cum_sq_diff = 0.0

            diff = close[i] - vwap[i]
            cum_vol += volume[i]
            cum_sq_diff += (diff ** 2) * volume[i]

            if cum_vol > 0:
                variance = cum_sq_diff / cum_vol
                std_dev[i] = np.sqrt(variance)
            else:
                std_dev[i] = 0.0

        return std_dev

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate VWAP mean reversion signals.

        Returns:
            np.ndarray of int64: Signal for each bar (1=long, -1=short, 0=flat)

        Parameters from config:
          - vwap_type: 'session', 'weekly', or 'anchored'
          - entry_threshold: Std devs from VWAP to enter (default: 2.0)
          - exit_threshold: Std devs from VWAP to exit (default: 0.5)
          - min_volume: Minimum volume for entry (default: 0)
          - holding_bars: Max holding period (default: 30)
          - session_start_hour: For session VWAP (default: 9)
          - session_start_minute: For session VWAP (default: 30)
        """
        if volume_arr is None:
            raise ValueError("VWAP strategies require volume data")

        # Get parameters
        vwap_type = self.config.parameters.get("vwap_type", "session")
        entry_threshold = self.config.parameters.get("entry_threshold", 2.0)
        exit_threshold = self.config.parameters.get("exit_threshold", 0.5)
        min_volume = self.config.parameters.get("min_volume", 0)
        holding_bars = self.config.parameters.get("holding_bars", 30)
        session_start_hour = self.config.parameters.get("session_start_hour", 9)
        session_start_minute = self.config.parameters.get("session_start_minute", 30)

        n = len(close_arr)
        signals = np.zeros(n, dtype=np.int64)

        # Create anchor points based on vwap_type
        anchor_points = np.zeros(n, dtype=bool)

        if vwap_type == "session":
            # Anchor at session start (assumes 5-min bars, RTH 9:30-16:00)
            # For 5-min bars: 9:30 = bar 0, reset every 78 bars (6.5 hours)
            bars_per_session = 78  # 6.5 hours * 12 bars/hour
            for i in range(0, n, bars_per_session):
                anchor_points[i] = True

        elif vwap_type == "weekly":
            # Anchor at weekly start (Monday 9:30)
            # For 5-min bars: ~1950 bars per week (5 days * 6.5 hours * 12 bars/hour)
            bars_per_week = 1950
            for i in range(0, n, bars_per_week):
                anchor_points[i] = True

        elif vwap_type == "anchored":
            # Anchor at start only (single anchor for entire period)
            anchor_points[0] = True

        else:
            raise ValueError(f"Invalid vwap_type: {vwap_type}")

        # Calculate VWAP and standard deviation
        vwap = self._calculate_vwap(high_arr, low_arr, close_arr, volume_arr, anchor_points)
        std_dev = self._calculate_std_dev(close_arr, vwap, volume_arr, anchor_points)

        # Calculate deviation in standard deviations
        deviation = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if std_dev[i] > 0:
                deviation[i] = (close_arr[i] - vwap[i]) / std_dev[i]
            else:
                deviation[i] = 0.0

        # Track position state
        position = 0
        entry_bar = 0
        entry_deviation = 0.0

        for i in range(n):
            # Skip if insufficient volume
            if volume_arr[i] < min_volume:
                signals[i] = position
                continue

            # Exit logic (check first)
            if position != 0:
                # Check time-based exit
                if i - entry_bar >= holding_bars:
                    position = 0
                    signals[i] = 0
                    continue

                # Check mean reversion exit
                if position == 1:  # Long (entered below VWAP)
                    # Exit when price reverts toward VWAP
                    if deviation[i] >= -exit_threshold:
                        position = 0
                        signals[i] = 0
                        continue

                elif position == -1:  # Short (entered above VWAP)
                    # Exit when price reverts toward VWAP
                    if deviation[i] <= exit_threshold:
                        position = 0
                        signals[i] = 0
                        continue

                # Still in position
                signals[i] = position
                continue

            # Entry logic
            if position == 0:
                # Long entry: price significantly below VWAP
                if deviation[i] <= -entry_threshold:
                    position = 1
                    entry_bar = i
                    entry_deviation = deviation[i]
                    signals[i] = 1
                    continue

                # Short entry: price significantly above VWAP
                elif deviation[i] >= entry_threshold:
                    position = -1
                    entry_bar = i
                    entry_deviation = deviation[i]
                    signals[i] = -1
                    continue

            # Flat
            signals[i] = 0

        return signals
