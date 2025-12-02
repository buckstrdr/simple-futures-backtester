"""Opening Range Breakout (ORB) Strategy.

Trades breakouts of the opening range (first N minutes of RTH session).
Classic intraday momentum strategy that exploits directional commitment
after initial price discovery period.

Logic:
  1. Define opening range: High/Low of first N minutes (5m, 15m, 30m)
  2. Wait for breakout above OR high or below OR low
  3. Enter on breakout with stop at opposite side of range
  4. Target: Risk/Reward ratio (1:1, 1:2, 1:3)
  5. Exit: Target, stop, or end of session

Different from trend-following:
  - Time-based (opening period specific)
  - Single trade per day maximum
  - Known risk (range size)
  - Directional commitment signal
"""
import numpy as np
from numpy.typing import NDArray
from simple_futures_backtester.strategy.base import BaseStrategy


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy with configurable OR window and R:R."""

    def generate_signals(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Generate ORB signals.

        Returns:
            np.ndarray of int64: Signal for each bar (1=long, -1=short, 0=flat)

        Parameters from config:
          - or_minutes: Opening range window in minutes (5, 15, 30)
          - rr_ratio: Risk/Reward ratio (1.0, 2.0, 3.0)
          - bars_per_session: Number of bars per session (5m = 78 bars, 15m = 26 bars)
          - breakout_buffer: Price buffer above/below OR to trigger entry (avoid false breaks)
          - min_range_pct: Minimum range size as % of price (avoid small ranges)
        """
        # Get parameters
        or_minutes = self.config.parameters.get("or_minutes", 30)
        rr_ratio = self.config.parameters.get("rr_ratio", 2.0)
        bars_per_session = self.config.parameters.get("bars_per_session", 78)
        breakout_buffer = self.config.parameters.get("breakout_buffer", 0.0005)  # 0.05%
        min_range_pct = self.config.parameters.get("min_range_pct", 0.0010)  # Min 0.10% range
        bar_freq = self.config.parameters.get("bar_freq_minutes", 5)

        # Calculate bars per opening range
        or_bars = or_minutes // bar_freq

        n = len(close_arr)
        signals = np.zeros(n, dtype=np.int64)

        # Track state per session
        position = 0  # 0=flat, 1=long, -1=short
        or_high = 0.0
        or_low = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        range_valid = False

        for i in range(n):
            # Determine which session we're in
            bar_in_session = i % bars_per_session

            # New session started
            if bar_in_session == 0:
                position = 0
                signals[i] = 0
                or_high = high_arr[i]
                or_low = low_arr[i]
                range_valid = False
                continue

            # Building opening range (first or_bars of session)
            if bar_in_session < or_bars:
                or_high = max(or_high, high_arr[i])
                or_low = min(or_low, low_arr[i])
                signals[i] = position
                continue

            # Opening range complete - validate
            if bar_in_session == or_bars and not range_valid:
                range_size = or_high - or_low
                range_pct = range_size / or_low if or_low > 0 else 0
                range_valid = (range_pct >= min_range_pct)

            # If range invalid, stay flat for rest of session
            if not range_valid:
                signals[i] = 0
                position = 0
                continue

            # Already in position - check exit
            if position != 0:
                # Check stop loss
                if position == 1:  # Long
                    if low_arr[i] <= stop_loss:
                        position = 0
                        signals[i] = 0
                        continue
                    # Check take profit
                    if high_arr[i] >= take_profit:
                        position = 0
                        signals[i] = 0
                        continue

                elif position == -1:  # Short
                    if high_arr[i] >= stop_loss:
                        position = 0
                        signals[i] = 0
                        continue
                    # Check take profit
                    if low_arr[i] <= take_profit:
                        position = 0
                        signals[i] = 0
                        continue

                # Check end of session exit (last 5 bars)
                if bar_in_session >= bars_per_session - 5:
                    position = 0
                    signals[i] = 0
                    continue

                # Still in position
                signals[i] = position
                continue

            # Not in position - check for breakout entry
            if position == 0 and bar_in_session >= or_bars:
                # Long breakout
                breakout_long_level = or_high * (1 + breakout_buffer)
                if high_arr[i] > breakout_long_level:
                    entry_price = breakout_long_level
                    stop_loss = or_low
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * rr_ratio)
                    position = 1
                    signals[i] = 1
                    continue

                # Short breakout
                breakout_short_level = or_low * (1 - breakout_buffer)
                if low_arr[i] < breakout_short_level:
                    entry_price = breakout_short_level
                    stop_loss = or_high
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * rr_ratio)
                    position = -1
                    signals[i] = -1
                    continue

            # Flat
            signals[i] = 0

        return signals
