"""MNQ 1m CHOP/ADX Trend-Pullback Scalper Strategy.

A trend-following scalping strategy for MNQ (Micro Nasdaq) on 1-minute charts that
identifies strong trends using CHOP and ADX filters, waits for pullbacks to Keltner
Channel levels, then enters when price resumes the trend.

Strategy Components:
- Choppiness Index (CHOP) for trend vs sideways detection
- ADX with +DI/-DI for trend strength and direction
- Keltner Channels for pullback entry zones
- Fast EMA(9) and Slow EMA(34) for trend confirmation
- ATR-based TP/SL and volatility measurement
- CHOP kill-switch for emergency exits

Key Features:
- Only trades when market is trending (CHOP < 38 AND ADX > 20)
- Dual directional filters (+DI/-DI) for clear trend direction
- Pullback entries to Keltner bands for better risk/reward
- CHOP-based emergency exit when trend decays
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


def calculate_ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Calculate Exponential Moving Average.

    Args:
        data: Input data array
        period: EMA period

    Returns:
        EMA values
    """
    n = len(data)
    ema = np.full(n, np.nan, dtype=np.float64)

    # Use SMA for first value
    if n >= period:
        ema[period - 1] = np.mean(data[:period])

        # Calculate EMA
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema


def calculate_atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    n = len(close)

    # Calculate True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = np.nan

    # Calculate ATR as EMA of TR
    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        atr[period - 1] = np.nanmean(tr[:period])

        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            atr[i] = (tr[i] - atr[i-1]) * multiplier + atr[i-1]

    return atr


def calculate_chop(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """Calculate Choppiness Index (CHOP).

    CHOP measures if market is trending or sideways:
    - Low values (< 38) = trending
    - High values (> 61) = choppy/sideways

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CHOP period

    Returns:
        CHOP values (0-100)
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)

    # Calculate True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = np.nan

    log_period = np.log10(period)

    for i in range(period, n):
        atr_sum = np.nansum(tr[i - period + 1 : i + 1])
        highest_high = np.nanmax(high[i - period + 1 : i + 1])
        lowest_low = np.nanmin(low[i - period + 1 : i + 1])

        price_range = highest_high - lowest_low

        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / log_period

    return chop


def calculate_adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate ADX (Average Directional Index) with +DI and -DI.

    ADX measures trend strength (not direction):
    - ADX < 20 = weak/no trend
    - ADX > 20 = trend present
    - ADX > 40 = strong trend

    +DI and -DI indicate trend direction:
    - +DI > -DI = uptrend
    - -DI > +DI = downtrend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        Tuple of (ADX, +DI, -DI) arrays
    """
    n = len(close)

    # Calculate directional movement
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)

    # Remove DM where opposite move is larger
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)

    plus_dm[0] = np.nan
    minus_dm[0] = np.nan

    # Calculate True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = np.nan

    # Smooth DM and TR using Wilder's smoothing (similar to EMA)
    smoothed_plus_dm = np.full(n, np.nan, dtype=np.float64)
    smoothed_minus_dm = np.full(n, np.nan, dtype=np.float64)
    smoothed_tr = np.full(n, np.nan, dtype=np.float64)

    if n >= period:
        # Initialize with sum
        smoothed_plus_dm[period - 1] = np.nansum(plus_dm[:period])
        smoothed_minus_dm[period - 1] = np.nansum(minus_dm[:period])
        smoothed_tr[period - 1] = np.nansum(tr[:period])

        # Wilder's smoothing
        for i in range(period, n):
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]
            smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]

    # Calculate directional indicators
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)

    valid = smoothed_tr > 0
    plus_di[valid] = 100.0 * smoothed_plus_dm[valid] / smoothed_tr[valid]
    minus_di[valid] = 100.0 * smoothed_minus_dm[valid] / smoothed_tr[valid]

    # Calculate DX and ADX
    dx = np.full(n, np.nan, dtype=np.float64)
    di_sum = plus_di + minus_di
    valid = di_sum > 0
    dx[valid] = 100.0 * np.abs(plus_di[valid] - minus_di[valid]) / di_sum[valid]

    # ADX is smoothed DX
    adx = np.full(n, np.nan, dtype=np.float64)
    if n >= period * 2:
        # Initialize ADX with average of first period DX values
        adx[period * 2 - 1] = np.nanmean(dx[period:period*2])

        # Smooth ADX
        for i in range(period * 2, n):
            adx[i] = ((period - 1) * adx[i-1] + dx[i]) / period

    return adx, plus_di, minus_di


def calculate_keltner_channels(
    close: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    ema_period: int,
    atr_period: int,
    atr_mult: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate Keltner Channels.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        ema_period: EMA period for middle line
        atr_period: ATR period for bands
        atr_mult: ATR multiplier for bands

    Returns:
        Tuple of (middle, upper, lower) bands
    """
    middle = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)

    upper = middle + (atr * atr_mult)
    lower = middle - (atr * atr_mult)

    return middle, upper, lower


class MNQChopADXScalper(BaseStrategy):
    """MNQ 1m CHOP/ADX Trend-Pullback Scalper Strategy.

    A trend-following scalping strategy that uses CHOP and ADX filters to identify
    strong trends, waits for pullbacks to Keltner Channels, then enters when price
    resumes the trend.

    Parameters (from StrategyConfig.parameters):
        chop_period: Period for Choppiness Index (default: 14)
        adx_period: Period for ADX calculation (default: 14)
        fast_ema_period: Period for fast EMA (default: 9)
        slow_ema_period: Period for slow EMA (default: 34)
        atr_period: Period for ATR calculation (default: 20)
        keltner_atr_mult: ATR multiplier for Keltner bands (default: 1.5)
        chop_threshold: CHOP threshold for trending market (default: 38)
        adx_threshold: ADX threshold for trend strength (default: 20)
        chop_exit_threshold: CHOP threshold for emergency exit (default: 50)
        pullback_buffer: ATR multiplier for pullback zone (default: 0.3)
        tp_multiplier: ATR multiplier for take profit (default: 1.8)
        sl_multiplier: ATR multiplier for stop loss (default: 1.0)
        swing_lookback: Bars to check for swing high/low (default: 5)
        use_swing_filter: Enable swing high/low filter (default: False)
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize MNQ CHOP/ADX Scalper strategy.

        Args:
            config: StrategyConfig containing strategy name and parameters.
        """
        super().__init__(config)
        self.chop_period: int = self.config.parameters.get("chop_period", 14)
        self.adx_period: int = self.config.parameters.get("adx_period", 14)
        self.fast_ema_period: int = self.config.parameters.get("fast_ema_period", 9)
        self.slow_ema_period: int = self.config.parameters.get("slow_ema_period", 34)
        self.atr_period: int = self.config.parameters.get("atr_period", 20)
        self.keltner_atr_mult: float = self.config.parameters.get("keltner_atr_mult", 1.5)
        self.chop_threshold: float = self.config.parameters.get("chop_threshold", 38.0)
        self.adx_threshold: float = self.config.parameters.get("adx_threshold", 20.0)
        self.chop_exit_threshold: float = self.config.parameters.get("chop_exit_threshold", 50.0)
        self.pullback_buffer: float = self.config.parameters.get("pullback_buffer", 0.3)
        self.tp_multiplier: float = self.config.parameters.get("tp_multiplier", 1.8)
        self.sl_multiplier: float = self.config.parameters.get("sl_multiplier", 1.0)
        self.swing_lookback: int = self.config.parameters.get("swing_lookback", 5)
        self.use_swing_filter: bool = self.config.parameters.get("use_swing_filter", False)

    def generate_signals(
        self,
        open_prices: NDArray[np.float64],
        high_prices: NDArray[np.float64],
        low_prices: NDArray[np.float64],
        close_prices: NDArray[np.float64],
        volume: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int32]:
        """Generate trading signals based on CHOP/ADX regime and Keltner pullbacks.

        Signal values:
        - 1: Long entry
        - -1: Short entry
        - 0: No position/exit

        Args:
            open_prices: Open prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Close prices
            volume: Volume (optional, not used)

        Returns:
            Array of signals (1 = long, -1 = short, 0 = no position)
        """
        n = len(close_prices)
        signals = np.zeros(n, dtype=np.int32)

        # Calculate all indicators
        chop = calculate_chop(high_prices, low_prices, close_prices, self.chop_period)
        adx, plus_di, minus_di = calculate_adx(high_prices, low_prices, close_prices, self.adx_period)
        fast_ema = calculate_ema(close_prices, self.fast_ema_period)
        slow_ema = calculate_ema(close_prices, self.slow_ema_period)
        atr = calculate_atr(high_prices, low_prices, close_prices, self.atr_period)
        keltner_mid, keltner_upper, keltner_lower = calculate_keltner_channels(
            close_prices, high_prices, low_prices,
            self.slow_ema_period, self.atr_period, self.keltner_atr_mult
        )

        # Track position state
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0

        # Start after enough bars for all indicators
        start_idx = max(self.chop_period, self.adx_period * 2, self.slow_ema_period, self.atr_period) + self.swing_lookback

        for i in range(start_idx, n):
            # Skip if any indicator is NaN
            if (np.isnan(chop[i]) or np.isnan(adx[i]) or np.isnan(plus_di[i]) or
                np.isnan(minus_di[i]) or np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]) or
                np.isnan(atr[i]) or np.isnan(keltner_mid[i])):
                continue

            # Check if in position
            if position != 0:
                # Emergency exit: CHOP kill-switch
                if chop[i] > self.chop_exit_threshold:
                    signals[i] = 0
                    position = 0
                    continue

                # Check TP/SL
                if position == 1:
                    if high_prices[i] >= tp_price:
                        signals[i] = 0
                        position = 0
                        continue
                    elif low_prices[i] <= sl_price:
                        signals[i] = 0
                        position = 0
                        continue
                elif position == -1:
                    if low_prices[i] <= tp_price:
                        signals[i] = 0
                        position = 0
                        continue
                    elif high_prices[i] >= sl_price:
                        signals[i] = 0
                        position = 0
                        continue

                # Stay in position
                signals[i] = position
                continue

            # No position - look for entry

            # Step 1: Check regime filters
            trending = (chop[i] < self.chop_threshold) and (adx[i] > self.adx_threshold)
            if not trending:
                continue

            # Step 2: Determine directional regime
            bullish_regime = (plus_di[i] > minus_di[i]) and (close_prices[i] > slow_ema[i])
            bearish_regime = (minus_di[i] > plus_di[i]) and (close_prices[i] < slow_ema[i])

            if not bullish_regime and not bearish_regime:
                continue

            # Step 3: Check for pullback and entry trigger
            if bullish_regime:
                # Check if price pulled back to Keltner bands
                distance_to_lower = low_prices[i] - keltner_lower[i]
                distance_to_mid = abs(low_prices[i] - keltner_mid[i])
                pullback_zone = self.pullback_buffer * atr[i]

                near_lower = distance_to_lower <= pullback_zone
                near_mid = distance_to_mid <= pullback_zone

                if near_lower or near_mid:
                    # Check entry trigger
                    bullish_candle = close_prices[i] > open_prices[i]
                    above_fast_ema = close_prices[i] > fast_ema[i]
                    above_keltner_mid = close_prices[i] > keltner_mid[i]

                    # Optional swing filter
                    swing_ok = True
                    if self.use_swing_filter and i >= self.swing_lookback:
                        swing_low = np.min(low_prices[i - self.swing_lookback : i])
                        swing_ok = low_prices[i] > swing_low

                    if bullish_candle and above_fast_ema and above_keltner_mid and swing_ok:
                        # Enter long
                        signals[i] = 1
                        position = 1
                        entry_price = close_prices[i]
                        tp_price = entry_price + (self.tp_multiplier * atr[i])
                        sl_price = entry_price - (self.sl_multiplier * atr[i])

            elif bearish_regime:
                # Check if price pulled back to Keltner bands
                distance_to_upper = keltner_upper[i] - high_prices[i]
                distance_to_mid = abs(high_prices[i] - keltner_mid[i])
                pullback_zone = self.pullback_buffer * atr[i]

                near_upper = distance_to_upper <= pullback_zone
                near_mid = distance_to_mid <= pullback_zone

                if near_upper or near_mid:
                    # Check entry trigger
                    bearish_candle = close_prices[i] < open_prices[i]
                    below_fast_ema = close_prices[i] < fast_ema[i]
                    below_keltner_mid = close_prices[i] < keltner_mid[i]

                    # Optional swing filter
                    swing_ok = True
                    if self.use_swing_filter and i >= self.swing_lookback:
                        swing_high = np.max(high_prices[i - self.swing_lookback : i])
                        swing_ok = high_prices[i] < swing_high

                    if bearish_candle and below_fast_ema and below_keltner_mid and swing_ok:
                        # Enter short
                        signals[i] = -1
                        position = -1
                        entry_price = close_prices[i]
                        tp_price = entry_price - (self.tp_multiplier * atr[i])
                        sl_price = entry_price + (self.sl_multiplier * atr[i])

        return signals


__all__ = ["MNQChopADXScalper"]
