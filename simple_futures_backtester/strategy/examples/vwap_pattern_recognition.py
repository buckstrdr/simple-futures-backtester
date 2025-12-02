"""VWAP Pattern Recognition Strategy - MGC Style.

Based on observed behavioral patterns around VWAP:

Pattern 1: Extension-Reversion
  - Price extends to 2SD from VWAP
  - Then reverts back toward VWAP (to ~0.25SD zone)
  - This is the "mean reversion" phase

Pattern 2: Consolidation-Continuation
  - Price consolidates in 0.25SD zone (near VWAP)
  - When next 5-min candle closes OUTSIDE this zone
  - Price likely continues to 1.5-2SD (directional move)
  - This is the "continuation" phase

Key Insight: VWAP acts as equilibrium. Price oscillates around it in
predictable ways:
  - Far from VWAP (>2SD) → mean reversion likely
  - Near VWAP (<0.25SD) + breakout → continuation likely

Different from simple mean reversion:
  - Uses pattern recognition (two-stage behavior)
  - Identifies regime shifts (reversion vs continuation)
  - Context-dependent entry (depends on recent price action)
"""
import numpy as np
from numpy.typing import NDArray
from simple_futures_backtester.strategy.base import BaseStrategy


class VWAPPatternRecognitionStrategy(BaseStrategy):
    """VWAP pattern recognition with extension-reversion and continuation patterns."""

    def _calculate_vwap(
        self,
        high: NDArray,
        low: NDArray,
        close: NDArray,
        volume: NDArray,
        anchor_points: NDArray,
    ) -> NDArray:
        """Calculate VWAP with anchoring support."""
        n = len(close)
        vwap = np.zeros(n, dtype=np.float64)
        typical_price = (high + low + close) / 3

        cum_vol = 0.0
        cum_tp_vol = 0.0

        for i in range(n):
            if anchor_points[i]:
                cum_vol = 0.0
                cum_tp_vol = 0.0

            cum_vol += volume[i]
            cum_tp_vol += typical_price[i] * volume[i]

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
        """Generate VWAP pattern recognition signals.

        Returns:
            np.ndarray of int64: Signal for each bar (1=long, -1=short, 0=flat)

        Parameters from config:
          - vwap_type: 'session', 'weekly', or 'anchored' (default: 'session')
          - extension_threshold: SD threshold for "extended" (default: 2.0)
          - reversion_zone: SD threshold for "near VWAP" zone (default: 0.25)
          - continuation_target_min: Min SD target for continuation (default: 1.5)
          - continuation_target_max: Max SD target for continuation (default: 2.0)
          - holding_bars: Max holding period (default: 20)
          - require_extension: Require recent extension before continuation trade (default: True)
          - extension_lookback: Bars to lookback for extension pattern (default: 10)
        """
        if volume_arr is None:
            raise ValueError("VWAP strategies require volume data")

        # Get parameters
        vwap_type = self.config.parameters.get("vwap_type", "session")
        extension_threshold = self.config.parameters.get("extension_threshold", 2.0)
        reversion_zone = self.config.parameters.get("reversion_zone", 0.25)
        continuation_target_min = self.config.parameters.get("continuation_target_min", 1.5)
        continuation_target_max = self.config.parameters.get("continuation_target_max", 2.0)
        holding_bars = self.config.parameters.get("holding_bars", 20)
        require_extension = self.config.parameters.get("require_extension", True)
        extension_lookback = self.config.parameters.get("extension_lookback", 10)

        n = len(close_arr)
        signals = np.zeros(n, dtype=np.int64)

        # Create anchor points based on vwap_type
        anchor_points = np.zeros(n, dtype=bool)

        if vwap_type == "session":
            bars_per_session = 78  # 6.5 hours * 12 bars/hour (5-min bars)
            for i in range(0, n, bars_per_session):
                anchor_points[i] = True
        elif vwap_type == "weekly":
            bars_per_week = 1950
            for i in range(0, n, bars_per_week):
                anchor_points[i] = True
        elif vwap_type == "anchored":
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

        # Track position and pattern state
        position = 0
        entry_bar = 0
        entry_price = 0.0
        target_sd = 0.0
        stop_sd = 0.0

        for i in range(extension_lookback, n):
            # Exit logic (check first)
            if position != 0:
                # Time-based exit
                if i - entry_bar >= holding_bars:
                    position = 0
                    signals[i] = 0
                    continue

                # Target/stop exit
                current_sd = deviation[i]

                if position == 1:  # Long
                    # Target: price reaches target_sd
                    if current_sd >= target_sd:
                        position = 0
                        signals[i] = 0
                        continue
                    # Stop: price goes against us (crosses back below VWAP significantly)
                    if current_sd < stop_sd:
                        position = 0
                        signals[i] = 0
                        continue

                elif position == -1:  # Short
                    # Target: price reaches target_sd (negative)
                    if current_sd <= -target_sd:
                        position = 0
                        signals[i] = 0
                        continue
                    # Stop: price goes against us (crosses back above VWAP significantly)
                    if current_sd > -stop_sd:
                        position = 0
                        signals[i] = 0
                        continue

                # Still in position
                signals[i] = position
                continue

            # Entry logic - Pattern Recognition
            current_sd = deviation[i]
            prev_sd = deviation[i-1]

            # Check for recent extension in lookback window
            recent_extended_long = False
            recent_extended_short = False

            if require_extension:
                for j in range(1, min(extension_lookback + 1, i)):
                    if deviation[i-j] >= extension_threshold:
                        recent_extended_long = True
                    if deviation[i-j] <= -extension_threshold:
                        recent_extended_short = True

            # PATTERN 1: Extension-Reversion (Mean Reversion Trade)
            # Price was extended (>2SD), now reverting back toward VWAP
            # Entry: when price crosses back INTO reversion zone from extended zone

            # Long: Price was below -extension_threshold, now crossing back up into reversion zone
            if prev_sd <= -extension_threshold and current_sd > -extension_threshold:
                # Mean reversion long (price bouncing back up toward VWAP)
                position = 1
                entry_bar = i
                entry_price = close_arr[i]
                target_sd = reversion_zone  # Target: get back to near VWAP
                stop_sd = -(extension_threshold + 0.5)  # Stop: further extension
                signals[i] = 1
                continue

            # Short: Price was above +extension_threshold, now crossing back down into reversion zone
            if prev_sd >= extension_threshold and current_sd < extension_threshold:
                # Mean reversion short (price pulling back down toward VWAP)
                position = -1
                entry_bar = i
                entry_price = close_arr[i]
                target_sd = reversion_zone  # Target: get back to near VWAP
                stop_sd = -(extension_threshold + 0.5)  # Stop: further extension
                signals[i] = -1
                continue

            # PATTERN 2: Consolidation-Continuation (Breakout from VWAP zone)
            # Price was consolidating near VWAP (<0.25SD), now breaking out
            # Entry: when price closes OUTSIDE reversion zone after being inside

            # Long continuation: Was in reversion zone, now breaking above
            if (abs(prev_sd) <= reversion_zone and current_sd > reversion_zone):
                # Only trade if we had recent upside extension (establishes uptrend context)
                if not require_extension or recent_extended_long:
                    position = 1
                    entry_bar = i
                    entry_price = close_arr[i]
                    target_sd = continuation_target_max  # Target: 1.5-2SD
                    stop_sd = -reversion_zone  # Stop: back below VWAP zone
                    signals[i] = 1
                    continue

            # Short continuation: Was in reversion zone, now breaking below
            if (abs(prev_sd) <= reversion_zone and current_sd < -reversion_zone):
                # Only trade if we had recent downside extension (establishes downtrend context)
                if not require_extension or recent_extended_short:
                    position = -1
                    entry_bar = i
                    entry_price = close_arr[i]
                    target_sd = continuation_target_max  # Target: 1.5-2SD
                    stop_sd = -reversion_zone  # Stop: back above VWAP zone
                    signals[i] = -1
                    continue

            # Flat
            signals[i] = 0

        return signals
