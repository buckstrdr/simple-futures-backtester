"""Supertrend + ADX + Volume Confluence Strategy.

Trend-following system using Supertrend for direction and entries, ADX for
trend strength filtering, and volume for momentum confirmation.

Core Logic:
    - Supertrend: Trend direction + dynamic trailing SL
    - ADX: Filters for strong trends only (ADX > threshold)
    - +DI/-DI: Directional momentum confirmation
    - Volume: Confirms participation (above average)

Entry Logic:
    LONG: Supertrend flips bullish OR price pulls back to Supertrend +
          ADX > threshold + +DI > -DI + Volume > avg
    SHORT: Supertrend flips bearish OR price pulls back to Supertrend +
           ADX > threshold + -DI > +DI + Volume > avg

Exit Logic:
    - TP1: tp1_atr_mult × ATR (50% position)
    - TP2: tp2_atr_mult × ATR (remaining 50%)
    - SL: Supertrend line - sl_buffer × ATR (dynamic trailing)
    - Supertrend flip exit (immediate)
    - ADX decay exit (trend dying)
    - Optional DI cross exit

Parameters:
    st_period: Supertrend ATR period (default: 10)
    st_multiplier: Supertrend band multiplier (default: 3.0)
    pullback_buffer: % buffer for pullback entries (default: 0.001)
    adx_period: ADX calculation period (default: 14)
    adx_threshold: Min ADX for entry (default: 20)
    adx_exit_threshold: ADX level to exit (default: 15)
    volume_period: Volume SMA period (default: 20)
    vol_mult: Volume multiplier threshold (default: 1.0)
    tp1_atr_mult: TP1 in ATR multiples (default: 1.5)
    tp2_atr_mult: TP2 in ATR multiples (default: 3.0)
    sl_buffer: ATR buffer below Supertrend for SL (default: 0.2)
    use_pullback_entries: Enable pullback entries (default: True)
    use_di_filter: Require +DI/-DI confirmation (default: True)
    use_di_exit: Exit on DI cross (default: False)

Example YAML config:
    strategy:
      name: supertrend_adx_volume
      parameters:
        st_period: 10
        st_multiplier: 3.0
        adx_threshold: 20
        vol_mult: 1.0
        tp1_atr_mult: 1.5
        tp2_atr_mult: 3.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


class SupertrendADXVolumeStrategy(BaseStrategy):
    """Supertrend + ADX + Volume confluence trend-following strategy.

    Combines Supertrend for direction/entries, ADX for trend strength filtering,
    and volume for participation confirmation. Uses dynamic trailing stops.

    Key Features:
        - Supertrend dynamic trailing stops
        - ADX filters weak trends
        - Volume confirms momentum
        - Two entry modes: flip only or flip + pullback
        - Multiple exit conditions (TP, SL, ST flip, ADX decay, DI cross)
        - Partial exits for profit management

    Attributes:
        st_period: Supertrend ATR period
        st_multiplier: Supertrend band multiplier
        pullback_buffer: Pullback entry buffer percentage
        adx_period: ADX calculation period
        adx_threshold: Minimum ADX for entries
        adx_exit_threshold: ADX threshold for exits
        volume_period: Volume SMA period
        vol_mult: Volume multiplier threshold
        tp1_atr_mult: TP1 ATR multiplier
        tp2_atr_mult: TP2 ATR multiplier
        sl_buffer: Stop loss ATR buffer
        use_pullback_entries: Enable pullback entries
        use_di_filter: Require DI confirmation
        use_di_exit: Enable DI cross exits
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize SupertrendADXVolumeStrategy with configuration.

        Args:
            config: StrategyConfig containing strategy name and parameters.
        """
        super().__init__(config)
        self.st_period: int = self.config.parameters.get("st_period", 10)
        self.st_multiplier: float = self.config.parameters.get("st_multiplier", 3.0)
        self.pullback_buffer: float = self.config.parameters.get("pullback_buffer", 0.001)
        self.adx_period: int = self.config.parameters.get("adx_period", 14)
        self.adx_threshold: float = self.config.parameters.get("adx_threshold", 20.0)
        self.adx_exit_threshold: float = self.config.parameters.get("adx_exit_threshold", 15.0)
        self.volume_period: int = self.config.parameters.get("volume_period", 20)
        self.vol_mult: float = self.config.parameters.get("vol_mult", 1.0)
        self.tp1_atr_mult: float = self.config.parameters.get("tp1_atr_mult", 1.5)
        self.tp2_atr_mult: float = self.config.parameters.get("tp2_atr_mult", 3.0)
        self.sl_buffer: float = self.config.parameters.get("sl_buffer", 0.2)
        self.use_pullback_entries: bool = self.config.parameters.get("use_pullback_entries", True)
        self.use_di_filter: bool = self.config.parameters.get("use_di_filter", True)
        self.use_di_exit: bool = self.config.parameters.get("use_di_exit", False)

    def calculate_indicators(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate Supertrend, ADX, +DI/-DI, Volume SMA, and ATR.

        Args:
            open_arr: Opening prices
            high_arr: High prices
            low_arr: Low prices
            close_arr: Closing prices
            volume_arr: Volume

        Returns:
            Dictionary with all calculated indicators
        """
        n = len(close_arr)

        # Calculate ATR
        tr = np.maximum(
            high_arr - low_arr,
            np.maximum(
                np.abs(high_arr - np.roll(close_arr, 1)),
                np.abs(low_arr - np.roll(close_arr, 1))
            )
        )
        tr[0] = high_arr[0] - low_arr[0]
        atr = self._ema(tr, self.st_period)

        # Supertrend calculation
        supertrend, st_direction = self._calculate_supertrend(
            high_arr, low_arr, close_arr, atr
        )

        # ADX with +DI/-DI
        adx, plus_di, minus_di = self._calculate_adx(
            high_arr, low_arr, close_arr, self.adx_period
        )

        # Volume SMA
        volume_sma = self._sma(volume_arr.astype(np.float64), self.volume_period)

        return {
            "supertrend": supertrend,
            "st_direction": st_direction,
            "atr": atr,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "volume_sma": volume_sma,
        }

    def _calculate_supertrend(
        self,
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        atr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Supertrend indicator.

        Args:
            high_arr: High prices
            low_arr: Low prices
            close_arr: Close prices
            atr: Average True Range

        Returns:
            Tuple of (supertrend_values, direction)
            direction: 1 = bullish, -1 = bearish
        """
        n = len(close_arr)
        hl2 = (high_arr + low_arr) / 2

        # Initial bands
        upper_band = hl2 + (self.st_multiplier * atr)
        lower_band = hl2 - (self.st_multiplier * atr)

        # Initialize arrays
        supertrend = np.zeros(n)
        direction = np.zeros(n)

        # First bar
        supertrend[0] = lower_band[0]
        direction[0] = 1

        for i in range(1, n):
            # Adjust lower band
            if lower_band[i] > lower_band[i-1] or close_arr[i-1] < lower_band[i-1]:
                final_lower = lower_band[i]
            else:
                final_lower = lower_band[i-1]

            # Adjust upper band
            if upper_band[i] < upper_band[i-1] or close_arr[i-1] > upper_band[i-1]:
                final_upper = upper_band[i]
            else:
                final_upper = upper_band[i-1]

            # Update bands
            lower_band[i] = final_lower
            upper_band[i] = final_upper

            # Determine direction
            if close_arr[i] > final_upper:
                direction[i] = 1  # Bullish
            elif close_arr[i] < final_lower:
                direction[i] = -1  # Bearish
            else:
                direction[i] = direction[i-1]

            # Set supertrend value
            if direction[i] == 1:
                supertrend[i] = final_lower
            else:
                supertrend[i] = final_upper

        return supertrend, direction

    def _calculate_adx(
        self,
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        period: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculate ADX, +DI, and -DI.

        Args:
            high_arr: High prices
            low_arr: Low prices
            close_arr: Close prices
            period: ADX period

        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        n = len(close_arr)

        # True Range
        tr = np.maximum(
            high_arr - low_arr,
            np.maximum(
                np.abs(high_arr - np.roll(close_arr, 1)),
                np.abs(low_arr - np.roll(close_arr, 1))
            )
        )
        tr[0] = high_arr[0] - low_arr[0]

        # Directional Movement
        high_diff = np.diff(high_arr, prepend=high_arr[0])
        low_diff = -np.diff(low_arr, prepend=low_arr[0])

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Smoothed values using EMA
        atr_smooth = self._ema(tr, period)
        plus_dm_smooth = self._ema(plus_dm, period)
        minus_dm_smooth = self._ema(minus_dm, period)

        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr_smooth)
        minus_di = 100 * (minus_dm_smooth / atr_smooth)

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)

        return adx, plus_di, minus_di

    def _ema(self, data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    def _sma(self, data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Calculate Simple Moving Average."""
        sma = np.zeros_like(data)
        for i in range(period-1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
        sma[:period-1] = data[:period-1]  # Fill early values
        return sma

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate Supertrend+ADX+Volume strategy signals.

        Entry Logic:
            LONG: (Supertrend flip bullish OR pullback to Supertrend) +
                  ADX > threshold + +DI > -DI + Volume > avg
            SHORT: (Supertrend flip bearish OR pullback to Supertrend) +
                   ADX > threshold + -DI > +DI + Volume > avg

        Exit Logic:
            - TP1/TP2 at ATR multiples
            - Dynamic trailing SL at Supertrend line
            - Supertrend flip exit
            - ADX decay exit
            - Optional DI cross exit

        Args:
            open_arr: Opening prices
            high_arr: High prices
            low_arr: Low prices
            close_arr: Closing prices
            volume_arr: Volume (required for this strategy)

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

        supertrend = indicators["supertrend"]
        st_direction = indicators["st_direction"]
        atr = indicators["atr"]
        adx = indicators["adx"]
        plus_di = indicators["plus_di"]
        minus_di = indicators["minus_di"]
        volume_sma = indicators["volume_sma"]

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

        for i in range(max(self.st_period, self.adx_period), n):
            # Check regime filters
            trending = adx[i] > self.adx_threshold
            volume_confirm = volume_arr[i] > (volume_sma[i] * self.vol_mult)

            # Exit logic (check first)
            if position != 0:
                # Supertrend flip exit
                if position == 1 and st_direction[i] == -1:
                    position = 0
                    signals[i] = 0
                    continue
                elif position == -1 and st_direction[i] == 1:
                    position = 0
                    signals[i] = 0
                    continue

                # ADX decay exit
                if adx[i] < self.adx_exit_threshold:
                    position = 0
                    signals[i] = 0
                    continue

                # DI cross exit (optional)
                if self.use_di_exit:
                    if position == 1 and minus_di[i] > plus_di[i]:
                        position = 0
                        signals[i] = 0
                        continue
                    elif position == -1 and plus_di[i] > minus_di[i]:
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

                # Update trailing stop
                if position == 1:
                    new_stop = supertrend[i] - (self.sl_buffer * atr[i])
                    stop_loss = max(stop_loss, new_stop)
                elif position == -1:
                    new_stop = supertrend[i] + (self.sl_buffer * atr[i])
                    stop_loss = min(stop_loss, new_stop)

                # Hold position
                signals[i] = position
                continue

            # Entry logic (only when flat)
            if not (trending and volume_confirm):
                signals[i] = 0
                continue

            # Long entry
            if st_direction[i] == 1:
                # Supertrend flip
                st_flip = (st_direction[i] == 1) and (st_direction[i-1] == -1)

                # Pullback to Supertrend
                st_pullback = False
                if self.use_pullback_entries:
                    st_pullback = low_arr[i] <= supertrend[i] * (1 + self.pullback_buffer)

                # DI confirmation
                di_confirm = True
                if self.use_di_filter:
                    di_confirm = plus_di[i] > minus_di[i]

                # Entry trigger
                entry_trigger = st_flip or st_pullback

                if entry_trigger and di_confirm:
                    position = 1
                    entry_price = close_arr[i]
                    entry_bar = i
                    tp1_hit = False

                    # Set stops and targets
                    stop_loss = supertrend[i] - (self.sl_buffer * atr[i])
                    tp1_level = entry_price + (self.tp1_atr_mult * atr[i])
                    tp2_level = entry_price + (self.tp2_atr_mult * atr[i])

                    signals[i] = 1
                    continue

            # Short entry
            if st_direction[i] == -1:
                # Supertrend flip
                st_flip = (st_direction[i] == -1) and (st_direction[i-1] == 1)

                # Pullback to Supertrend
                st_pullback = False
                if self.use_pullback_entries:
                    st_pullback = high_arr[i] >= supertrend[i] * (1 - self.pullback_buffer)

                # DI confirmation
                di_confirm = True
                if self.use_di_filter:
                    di_confirm = minus_di[i] > plus_di[i]

                # Entry trigger
                entry_trigger = st_flip or st_pullback

                if entry_trigger and di_confirm:
                    position = -1
                    entry_price = close_arr[i]
                    entry_bar = i
                    tp1_hit = False

                    # Set stops and targets
                    stop_loss = supertrend[i] + (self.sl_buffer * atr[i])
                    tp1_level = entry_price - (self.tp1_atr_mult * atr[i])
                    tp2_level = entry_price - (self.tp2_atr_mult * atr[i])

                    signals[i] = -1
                    continue

            signals[i] = 0

        return signals


__all__ = ["SupertrendADXVolumeStrategy"]
