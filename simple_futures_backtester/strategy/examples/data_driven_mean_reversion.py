"""Data-Driven Mean Reversion Strategy (MGC-style adapted for MNQ/MES/etc).

Based on empirically measured edges, not theoretical indicators.
Adapted from MGC Data-Driven Strategy document.

Core Patterns (tested on MGC, adapting for equity indices):
  1. Volume spike + down bar → LONG (exhaustion)
  2. Large bar (>2.5x ATR) + down → LONG (panic selling)
  3. Momentum washout (>0.3% drop in 5 bars) → LONG
  4. Consecutive down bars (5+) → LONG (mean reversion)
  5. Combined scoring system (confluence)

Key Insight: Markets fade selling exhaustion, not buying.
"""
import numpy as np
from numpy.typing import NDArray
from simple_futures_backtester.strategy.base import BaseStrategy


class DataDrivenMeanReversionStrategy(BaseStrategy):
    """Data-driven mean reversion using measured statistical edges."""

    def _calculate_atr(self, high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
        """Calculate ATR."""
        tr = np.maximum(high - low,
                        np.maximum(abs(high - np.roll(close, 1)),
                                   abs(low - np.roll(close, 1))))
        tr[:period] = high[:period] - low[:period]

        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _calculate_sma(self, data: NDArray, period: int) -> NDArray:
        """Simple moving average."""
        sma = np.full_like(data, np.nan, dtype=np.float64)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        return sma

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate mean reversion signals based on statistical edges.

        Returns:
            np.ndarray of int64: Signal for each bar (1=long, -1=short, 0=flat)

        Parameters from config:
          - vol_spike_threshold: Volume ratio to trigger (default: 3.0)
          - large_bar_threshold: Bar range vs ATR ratio (default: 2.5)
          - washout_pct: Momentum drop % threshold (default: 0.3)
          - lookback_bars: Bars for momentum calculation (default: 5)
          - tp_atr_mult: Take profit in ATR (default: 1.5)
          - sl_atr_mult: Stop loss in ATR (default: 1.0)
          - holding_bars: Max holding period (default: 10)
          - min_score: Min confluence score for entry (default: 2)
        """
        # Get parameters
        vol_spike_threshold = self.config.parameters.get("vol_spike_threshold", 3.0)
        large_bar_threshold = self.config.parameters.get("large_bar_threshold", 2.5)
        washout_pct = self.config.parameters.get("washout_pct", 0.3) / 100  # Convert to decimal
        lookback_bars = self.config.parameters.get("lookback_bars", 5)
        tp_atr_mult = self.config.parameters.get("tp_atr_mult", 1.5)
        sl_atr_mult = self.config.parameters.get("sl_atr_mult", 1.0)
        holding_bars = self.config.parameters.get("holding_bars", 10)
        min_score = self.config.parameters.get("min_score", 2)

        n = len(close_arr)
        signals = np.zeros(n, dtype=np.int64)

        # Calculate indicators
        atr = self._calculate_atr(high_arr, low_arr, close_arr, period=14)
        volume_sma = self._calculate_sma(volume_arr.astype(np.float64), period=20) if volume_arr is not None else np.ones(n)

        # Calculate bar characteristics
        bar_range = high_arr - low_arr
        bar_body = close_arr - open_arr
        is_down_bar = bar_body < 0

        # Track position state
        position = 0
        entry_price = 0.0
        entry_bar = 0
        stop_loss = 0.0
        take_profit = 0.0

        for i in range(max(lookback_bars, 20), n):
            # Exit logic (check first)
            if position != 0:
                # Check TP/SL
                if position == 1:  # Long
                    if low_arr[i] <= stop_loss:
                        position = 0
                        signals[i] = 0
                        continue
                    if high_arr[i] >= take_profit:
                        position = 0
                        signals[i] = 0
                        continue

                # Time-based exit
                if i - entry_bar >= holding_bars:
                    position = 0
                    signals[i] = 0
                    continue

                # Still in position
                signals[i] = position
                continue

            # Entry logic - calculate confluence score
            score = 0

            # 1. Volume spike on down bar (+2 points)
            if volume_arr is not None and volume_sma[i] > 0:
                volume_ratio = volume_arr[i] / volume_sma[i]
                if volume_ratio > vol_spike_threshold and is_down_bar[i]:
                    score += 2

            # 2. Large down bar (>2.5x ATR) (+1 point)
            if atr[i] > 0:
                range_ratio = bar_range[i] / atr[i]
                if range_ratio > large_bar_threshold and is_down_bar[i]:
                    score += 1

            # 3. Momentum washout (sharp drop) (+2 points)
            if i >= lookback_bars:
                momentum = (close_arr[i] - close_arr[i - lookback_bars]) / close_arr[i - lookback_bars]
                if momentum < -washout_pct:
                    score += 2

            # 4. Consecutive down bars (5+) (+1 point)
            if i >= 5:
                consecutive_down = all(is_down_bar[i-j] for j in range(5))
                if consecutive_down:
                    score += 1

            # Entry if score meets threshold
            if score >= min_score:
                position = 1
                entry_price = close_arr[i]
                entry_bar = i
                stop_loss = entry_price - (sl_atr_mult * atr[i])
                take_profit = entry_price + (tp_atr_mult * atr[i])
                signals[i] = 1

        return signals
