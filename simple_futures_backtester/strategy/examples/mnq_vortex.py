"""MNQ 1m Vortex Micro Scalper Strategy.

A scalping strategy for MNQ (Micro Nasdaq) on 1-minute charts that catches small
trend moves during periods of normal volatility by waiting for pullbacks to a moving
average, then entering when the trend resumes.

Strategy Components:
- Vortex Indicator (VI) for trend strength and direction
- Hull Moving Average (HMA) for trend identification and pullback detection
- ATR-based volatility filter and position sizing
- Trend flip emergency exits

Key Features:
- Volatility filter: Only trades during normal volatility (0.00025 < ATR/Price < 0.001)
- Regime detection: Long/Short/None based on VI and HMA slope
- Pullback entries: Waits for price to pull back to HMA before entering
- ATR-based TP/SL: Dynamic targets based on current volatility
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simple_futures_backtester.config import StrategyConfig
from simple_futures_backtester.strategy.base import BaseStrategy


def weighted_moving_average(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Calculate Weighted Moving Average.

    Args:
        data: Input data array
        period: WMA period

    Returns:
        WMA values
    """
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights = weights / weights.sum()

    result = np.full(len(data), np.nan, dtype=np.float64)
    for i in range(period - 1, len(data)):
        result[i] = np.dot(data[i - period + 1 : i + 1], weights)

    return result


def hull_moving_average(close: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Calculate Hull Moving Average.

    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    Args:
        close: Close prices
        period: HMA period (default: 21)

    Returns:
        HMA values
    """
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    # Step 1: WMA(n/2)
    wma_half = weighted_moving_average(close, half_period)

    # Step 2: WMA(n)
    wma_full = weighted_moving_average(close, period)

    # Step 3: Raw HMA = 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full

    # Step 4: HMA = WMA(Raw HMA, sqrt(n))
    hma = weighted_moving_average(raw_hma, sqrt_period)

    return hma


def vortex_indicator(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate Vortex Indicator (VI+ and VI-).

    VI measures trend strength and direction:
    - VI+ (positive vortex) = uptrend strength
    - VI- (negative vortex) = downtrend strength

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: VI period (default: 14)

    Returns:
        Tuple of (VI+, VI-) arrays
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
    tr[0] = np.nan  # First value is invalid

    # Calculate Vortex Movement
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan

    # Calculate rolling sums
    vi_plus = np.full(n, np.nan, dtype=np.float64)
    vi_minus = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        sum_vm_plus = np.nansum(vm_plus[i - period + 1 : i + 1])
        sum_vm_minus = np.nansum(vm_minus[i - period + 1 : i + 1])
        sum_tr = np.nansum(tr[i - period + 1 : i + 1])

        if sum_tr > 0:
            vi_plus[i] = sum_vm_plus / sum_tr
            vi_minus[i] = sum_vm_minus / sum_tr

    return vi_plus, vi_minus


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
        period: ATR period (default: 14)

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

    # Calculate ATR as rolling average of TR
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr[i] = np.nanmean(tr[i - period + 1 : i + 1])

    return atr


class MNQVortexScalper(BaseStrategy):
    """MNQ 1m Vortex Micro Scalper Strategy.

    A trend-following scalping strategy that uses Vortex Indicator and Hull Moving
    Average to identify trend regimes and pullback entry opportunities.

    Parameters (from StrategyConfig.parameters):
        vortex_period: Period for Vortex Indicator (default: 14)
        hma_period: Period for Hull Moving Average (default: 21)
        atr_period: Period for ATR calculation (default: 14)
        min_atr_per_bar: Minimum ATR/Price ratio for trading (default: 0.00025)
        max_atr_per_bar: Maximum ATR/Price ratio for trading (default: 0.0010)
        pullback_atr_mult: ATR multiplier for pullback distance (default: 0.5)
        tp_multiplier: ATR multiplier for take profit (default: 1.2)
        sl_multiplier: ATR multiplier for stop loss (default: 0.7)
        hma_slope_bars: Lookback bars for HMA slope calculation (default: 3)
        extension_limit_mult: Max ATR distance from HMA for entry (default: 2.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize MNQ Vortex Scalper strategy.

        Args:
            config: StrategyConfig containing strategy name and parameters.
        """
        super().__init__(config)
        self.vortex_period: int = self.config.parameters.get("vortex_period", 14)
        self.hma_period: int = self.config.parameters.get("hma_period", 21)
        self.atr_period: int = self.config.parameters.get("atr_period", 14)
        self.min_atr_per_bar: float = self.config.parameters.get("min_atr_per_bar", 0.00025)
        self.max_atr_per_bar: float = self.config.parameters.get("max_atr_per_bar", 0.0010)
        self.pullback_atr_mult: float = self.config.parameters.get("pullback_atr_mult", 0.5)
        self.tp_multiplier: float = self.config.parameters.get("tp_multiplier", 1.2)
        self.sl_multiplier: float = self.config.parameters.get("sl_multiplier", 0.7)
        self.hma_slope_bars: int = self.config.parameters.get("hma_slope_bars", 3)
        self.extension_limit_mult: float = self.config.parameters.get("extension_limit_mult", 2.0)

    def generate_signals(
        self,
        open_prices: NDArray[np.float64],
        high_prices: NDArray[np.float64],
        low_prices: NDArray[np.float64],
        close_prices: NDArray[np.float64],
        volume: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int32]:
        """Generate trading signals based on Vortex and HMA.

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

        # Calculate indicators
        vi_plus, vi_minus = vortex_indicator(
            high_prices, low_prices, close_prices, self.vortex_period
        )
        hma = hull_moving_average(close_prices, self.hma_period)
        atr = calculate_atr(high_prices, low_prices, close_prices, self.atr_period)

        # Calculate ATR per bar ratio for volatility filter
        atr_per_bar = atr / close_prices

        # Track position state
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0

        # Start after enough bars for all indicators
        start_idx = max(self.vortex_period, self.hma_period, self.atr_period) + self.hma_slope_bars

        for i in range(start_idx, n):
            # Skip if any indicator is NaN
            if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(hma[i]) or np.isnan(atr[i]):
                continue

            # Check if in position
            if position != 0:
                # Check exits
                # 1. TP hit
                if position == 1 and high_prices[i] >= tp_price:
                    signals[i] = 0
                    position = 0
                    continue
                elif position == -1 and low_prices[i] <= tp_price:
                    signals[i] = 0
                    position = 0
                    continue

                # 2. SL hit
                if position == 1 and low_prices[i] <= sl_price:
                    signals[i] = 0
                    position = 0
                    continue
                elif position == -1 and high_prices[i] >= sl_price:
                    signals[i] = 0
                    position = 0
                    continue

                # 3. Trend flip (VI cross) - DISABLED due to whipsaw on 1m timeframe
                # if position == 1 and vi_plus[i] < vi_minus[i]:
                #     signals[i] = 0
                #     position = 0
                #     continue
                # elif position == -1 and vi_minus[i] < vi_plus[i]:
                #     signals[i] = 0
                #     position = 0
                #     continue

                # Stay in position
                signals[i] = position
                continue

            # No position - look for entry

            # Step 1: Volatility filter
            if atr_per_bar[i] < self.min_atr_per_bar or atr_per_bar[i] > self.max_atr_per_bar:
                continue

            # Step 2: Identify regime
            hma_slope_up = hma[i] > hma[i - self.hma_slope_bars]
            hma_slope_down = hma[i] < hma[i - self.hma_slope_bars]

            long_regime = (vi_plus[i] > vi_minus[i]) and hma_slope_up
            short_regime = (vi_minus[i] > vi_plus[i]) and hma_slope_down

            if not long_regime and not short_regime:
                continue

            # Step 3: Check for pullback to HMA
            distance_to_hma = abs(close_prices[i] - hma[i])
            pullback_threshold = self.pullback_atr_mult * atr[i]

            if distance_to_hma > pullback_threshold:
                continue

            # Step 4: Check for trend resumption signal
            if long_regime:
                # Long entry conditions
                resumption = (
                    close_prices[i] > high_prices[i - 1] or
                    (close_prices[i] > open_prices[i] and close_prices[i] > hma[i])
                )

                # Extension filter
                too_extended = close_prices[i] > hma[i] + (self.extension_limit_mult * atr[i])

                if resumption and not too_extended:
                    # Enter long
                    signals[i] = 1
                    position = 1
                    entry_price = close_prices[i]
                    tp_price = entry_price + (self.tp_multiplier * atr[i])
                    sl_price = entry_price - (self.sl_multiplier * atr[i])

            elif short_regime:
                # Short entry conditions
                resumption = (
                    close_prices[i] < low_prices[i - 1] or
                    (close_prices[i] < open_prices[i] and close_prices[i] < hma[i])
                )

                # Extension filter
                too_extended = close_prices[i] < hma[i] - (self.extension_limit_mult * atr[i])

                if resumption and not too_extended:
                    # Enter short
                    signals[i] = -1
                    position = -1
                    entry_price = close_prices[i]
                    tp_price = entry_price - (self.tp_multiplier * atr[i])
                    sl_price = entry_price + (self.sl_multiplier * atr[i])

        return signals


__all__ = ["MNQVortexScalper"]
