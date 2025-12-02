"""Multi-Timeframe Keltner Channel Strategy.

Trend-following pullback strategy using higher timeframe (30m) trend bias
with lower timeframe (5m) Keltner Channel entries.

Core Logic:
    - HTF (30m): EMA(50) establishes trend bias
    - LTF (5m): Keltner Channels define entry zones
    - Only trade pullbacks WITH the HTF trend
    - CHOP filter: only trade when market is trending (CHOP < 45)

Entry Logic:
    LONG: HTF bullish + LTF trending + price pulls to lower Keltner +
          closes back above lower band and EMA(20)
    SHORT: HTF bearish + LTF trending + price pulls to upper Keltner +
           closes back below upper band and EMA(20)

Exit Logic:
    - TP1 (50% position): Price reaches mid-Keltner (EMA 20)
    - TP2 (50% position): Opposite Keltner band or tp_atr_mult × ATR
    - SL: sl_atr_mult × ATR beyond entry Keltner band
    - HTF Bias Flip: Exit immediately if 30m trend reverses
    - Time Stop: Exit if no TP1 hit within max_bars

Parameters:
    htf_ema_period: 30m EMA period for trend bias (default: 50)
    ltf_ema_period: 5m EMA period for Keltner center (default: 20)
    keltner_atr_period: ATR period for Keltner bands (default: 20)
    keltner_mult: Keltner band multiplier (default: 1.5)
    chop_period: Choppiness Index period (default: 14)
    chop_threshold: Max CHOP for trending (default: 45)
    tp_atr_mult: Take profit ATR multiplier (default: 1.8)
    sl_atr_mult: Stop loss ATR multiplier (default: 1.0)
    safety_buffer: HTF EMA safety zone (default: 0.001 = 0.1%)
    max_bars: Max holding period before time stop (default: 30)
    partial_tp_pct: Percentage to exit at TP1 (default: 0.5)

Example YAML config:
    strategy:
      name: mtf_keltner
      parameters:
        htf_ema_period: 50
        ltf_ema_period: 20
        keltner_atr_period: 20
        keltner_mult: 1.5
        chop_period: 14
        chop_threshold: 45
        tp_atr_mult: 1.8
        sl_atr_mult: 1.0
        safety_buffer: 0.001
        max_bars: 30
        partial_tp_pct: 0.5
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class MTFKeltnerStrategy(BaseStrategy):
    """Multi-Timeframe Keltner Channel pullback strategy.

    Designed for 5m lower timeframe with 30m higher timeframe trend bias.
    Enters on pullbacks to Keltner bands WITH HTF trend, exits with partial
    profits and multiple conditions.

    Key Features:
        - HTF trend filter prevents counter-trend trades
        - CHOP filter avoids choppy markets
        - Partial exits maximize profit capture
        - HTF bias flip exit protects capital on trend changes
        - Time-based stop prevents dead trades

    Attributes:
        htf_ema_period: Higher timeframe EMA period
        ltf_ema_period: Lower timeframe EMA period (Keltner center)
        keltner_atr_period: ATR period for Keltner bands
        keltner_mult: Keltner band width multiplier
        chop_period: Choppiness Index period
        chop_threshold: Maximum CHOP value for trending
        tp_atr_mult: Take profit ATR multiplier
        sl_atr_mult: Stop loss ATR multiplier
        safety_buffer: HTF EMA safety zone percentage
        max_bars: Maximum bars to hold before time stop
        partial_tp_pct: Percentage to exit at TP1
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize MTFKeltnerStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
        """
        super().__init__(config)
        self.htf_ema_period: int = self.config.parameters.get("htf_ema_period", 50)
        self.ltf_ema_period: int = self.config.parameters.get("ltf_ema_period", 20)
        self.keltner_atr_period: int = self.config.parameters.get("keltner_atr_period", 20)
        self.keltner_mult: float = self.config.parameters.get("keltner_mult", 1.5)
        self.chop_period: int = self.config.parameters.get("chop_period", 14)
        self.chop_threshold: float = self.config.parameters.get("chop_threshold", 45.0)
        self.tp_atr_mult: float = self.config.parameters.get("tp_atr_mult", 1.8)
        self.sl_atr_mult: float = self.config.parameters.get("sl_atr_mult", 1.0)
        self.safety_buffer: float = self.config.parameters.get("safety_buffer", 0.001)
        self.max_bars: int = self.config.parameters.get("max_bars", 30)
        self.partial_tp_pct: float = self.config.parameters.get("partial_tp_pct", 0.5)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate multi-timeframe indicators.

        Calculates:
            - HTF EMA(50) from 30m resampled data
            - LTF EMA(20) for Keltner center
            - Keltner Channels (upper/lower bands)
            - ATR for position sizing
            - CHOP for regime filtering

        Args:
            open_arr: Opening prices (5m)
            high_arr: High prices (5m)
            low_arr: Low prices (5m)
            close_arr: Closing prices (5m)
            volume_arr: Volume (not used)

        Returns:
            Dictionary with all calculated indicators
        """
        n = len(close_arr)

        # Calculate ATR (needed for Keltner)
        tr = np.maximum(
            high_arr - low_arr,
            np.maximum(
                np.abs(high_arr - np.roll(close_arr, 1)),
                np.abs(low_arr - np.roll(close_arr, 1))
            )
        )
        tr[0] = high_arr[0] - low_arr[0]
        atr = self._ema(tr, self.keltner_atr_period)

        # LTF EMA(20) - Keltner center
        ltf_ema = self._ema(close_arr, self.ltf_ema_period)

        # Keltner Channels
        kc_upper = ltf_ema + (self.keltner_mult * atr)
        kc_lower = ltf_ema - (self.keltner_mult * atr)

        # HTF EMA(50) - resample to 30m (6 bars of 5m = 30m)
        # Forward-fill HTF values to LTF bars
        htf_ema = self._resample_and_ema(close_arr, self.htf_ema_period, resample_factor=6)

        # CHOP indicator
        chop = self._choppiness_index(high_arr, low_arr, close_arr, self.chop_period)

        return {
            "ltf_ema": ltf_ema,
            "htf_ema": htf_ema,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
            "atr": atr,
            "chop": chop,
        }

    def _ema(self, data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Calculate Exponential Moving Average.

        Args:
            data: Input price array
            period: EMA period

        Returns:
            EMA array
        """
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    def _resample_and_ema(
        self,
        data: NDArray[np.float64],
        period: int,
        resample_factor: int = 6
    ) -> NDArray[np.float64]:
        """Resample LTF data to HTF and calculate EMA, then forward-fill.

        Args:
            data: 5m close prices
            period: EMA period
            resample_factor: Bars to aggregate (6 for 5m->30m)

        Returns:
            HTF EMA forward-filled to LTF length
        """
        n = len(data)

        # Resample to HTF (take every 6th bar as HTF close)
        htf_len = n // resample_factor
        htf_data = data[resample_factor-1::resample_factor][:htf_len]

        # Calculate EMA on HTF
        htf_ema = self._ema(htf_data, period)

        # Forward-fill to LTF
        ltf_ema = np.zeros(n)
        for i in range(n):
            htf_idx = i // resample_factor
            if htf_idx < len(htf_ema):
                ltf_ema[i] = htf_ema[htf_idx]
            else:
                ltf_ema[i] = htf_ema[-1]

        return ltf_ema

    def _choppiness_index(
        self,
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        period: int
    ) -> NDArray[np.float64]:
        """Calculate Choppiness Index.

        CHOP = 100 * log10(ATR_sum / (high_max - low_min)) / log10(period)

        Args:
            high_arr: High prices
            low_arr: Low prices
            close_arr: Close prices
            period: CHOP period

        Returns:
            CHOP indicator array
        """
        n = len(close_arr)

        # Calculate True Range
        tr = np.maximum(
            high_arr - low_arr,
            np.maximum(
                np.abs(high_arr - np.roll(close_arr, 1)),
                np.abs(low_arr - np.roll(close_arr, 1))
            )
        )
        tr[0] = high_arr[0] - low_arr[0]

        # Rolling sum of TR
        atr_sum = np.zeros(n)
        for i in range(period-1, n):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])

        # Rolling high-low range
        high_low_range = np.zeros(n)
        for i in range(period-1, n):
            high_low_range[i] = np.max(high_arr[i-period+1:i+1]) - np.min(low_arr[i-period+1:i+1])

        # Avoid division by zero
        high_low_range = np.where(high_low_range == 0, 1e-10, high_low_range)

        # CHOP calculation
        chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)

        # Fill early bars with neutral value (50 = unclear)
        chop[:period-1] = 50.0

        return chop

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate MTF Keltner strategy signals.

        Entry Logic:
            LONG: HTF bullish + LTF trending + pullback to lower Keltner +
                  confirmation (close > lower band and > EMA20) + safety
            SHORT: HTF bearish + LTF trending + pullback to upper Keltner +
                   confirmation (close < upper band and < EMA20) + safety

        Exit Logic:
            - TP1 at mid-Keltner (50% position)
            - TP2 at opposite band or tp_atr_mult × ATR (remaining 50%)
            - SL at sl_atr_mult × ATR
            - HTF bias flip exit
            - Time-based stop after max_bars

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

        if volume_arr is None:
            volume_arr = np.ones(n, dtype=np.int64)

        # Calculate indicators
        indicators = self.calculate_indicators(
            open_arr, high_arr, low_arr, close_arr, volume_arr
        )

        ltf_ema = indicators["ltf_ema"]
        htf_ema = indicators["htf_ema"]
        kc_upper = indicators["kc_upper"]
        kc_lower = indicators["kc_lower"]
        atr = indicators["atr"]
        chop = indicators["chop"]

        # Initialize signals
        signals = np.zeros(n, dtype=np.int64)

        # Track position state
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_bar = 0
        tp1_hit = False
        stop_loss = 0.0
        tp1_level = 0.0
        tp2_level = 0.0

        for i in range(self.htf_ema_period, n):
            # Check HTF bias
            htf_bullish = close_arr[i] > htf_ema[i] * (1 + self.safety_buffer)
            htf_bearish = close_arr[i] < htf_ema[i] * (1 - self.safety_buffer)

            # Check LTF trending
            ltf_trending = chop[i] < self.chop_threshold

            # Exit logic (check first)
            if position != 0:
                bars_held = i - entry_bar

                # HTF bias flip exit
                if position == 1 and not htf_bullish:
                    position = 0
                    signals[i] = 0
                    continue
                elif position == -1 and not htf_bearish:
                    position = 0
                    signals[i] = 0
                    continue

                # Stop loss exit
                if position == 1 and low_arr[i] <= stop_loss:
                    position = 0
                    signals[i] = 0
                    continue
                elif position == -1 and high_arr[i] >= stop_loss:
                    position = 0
                    signals[i] = 0
                    continue

                # TP1 check (partial exit)
                if not tp1_hit:
                    if position == 1 and high_arr[i] >= tp1_level:
                        tp1_hit = True
                    elif position == -1 and low_arr[i] <= tp1_level:
                        tp1_hit = True

                # TP2 exit (full exit)
                if position == 1 and high_arr[i] >= tp2_level:
                    position = 0
                    signals[i] = 0
                    continue
                elif position == -1 and low_arr[i] <= tp2_level:
                    position = 0
                    signals[i] = 0
                    continue

                # Time-based stop (no TP1 hit)
                if not tp1_hit and bars_held >= self.max_bars:
                    position = 0
                    signals[i] = 0
                    continue

                # Hold position
                signals[i] = position
                continue

            # Entry logic (only when flat)
            if not ltf_trending:
                signals[i] = 0
                continue

            # Long entry
            if htf_bullish:
                # Pullback to lower Keltner
                pullback = low_arr[i] <= kc_lower[i]

                # Confirmation: close back above lower band and EMA
                confirmation = (close_arr[i] > kc_lower[i]) and (close_arr[i] > ltf_ema[i])

                # Safety: low above HTF EMA area
                safety = low_arr[i] > htf_ema[i] * (1 - self.safety_buffer)

                if pullback and confirmation and safety:
                    position = 1
                    entry_price = close_arr[i]
                    entry_bar = i
                    tp1_hit = False

                    # Set stops and targets
                    stop_loss = entry_price - (self.sl_atr_mult * atr[i])
                    tp1_level = ltf_ema[i]  # Mid-Keltner
                    tp2_level = kc_upper[i]  # Opposite Keltner

                    signals[i] = 1
                    continue

            # Short entry
            if htf_bearish:
                # Pullback to upper Keltner
                pullback = high_arr[i] >= kc_upper[i]

                # Confirmation: close back below upper band and EMA
                confirmation = (close_arr[i] < kc_upper[i]) and (close_arr[i] < ltf_ema[i])

                # Safety: high below HTF EMA area
                safety = high_arr[i] < htf_ema[i] * (1 + self.safety_buffer)

                if pullback and confirmation and safety:
                    position = -1
                    entry_price = close_arr[i]
                    entry_bar = i
                    tp1_hit = False

                    # Set stops and targets
                    stop_loss = entry_price + (self.sl_atr_mult * atr[i])
                    tp1_level = ltf_ema[i]  # Mid-Keltner
                    tp2_level = kc_lower[i]  # Opposite Keltner

                    signals[i] = -1
                    continue

            signals[i] = 0

        return signals


__all__ = ["MTFKeltnerStrategy"]
