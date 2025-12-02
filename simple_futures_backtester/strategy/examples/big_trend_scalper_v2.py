"""Big Trend Double Check Strategy - Version 2 (Fixed Target).

Based on analysis showing that fixed ATR profit targets (+1.03% return)
matched the original indicator better than pure reversal detection.

Strategy Logic:
- Entry: Momentum breakout (8+ points) + Close > SMA(5) + ADX > 20
- Exit: Fixed profit target (1.0x ATR) OR trailing stop (1.5x ATR)
- The original indicator's "scalp exit" likely uses profit targets, not just reversals

Key Findings from Analysis:
- 71.2% win rate on scalp exits
- Average win: +20.23 points (0.98x ATR)
- Average bars held: 6-7 bars for winners
- Fixed ATR target version achieved +1.03% (close to original +0.97%)
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np


class BigTrendScalperV2Strategy:
    """Big Trend scalping strategy with fixed profit targets and trailing stops."""

    def __init__(
        self,
        # Entry parameters
        min_entry_momentum: float = 8.0,  # Points (from analysis: median 8.25)
        adx_period: int = 14,
        adx_threshold: float = 20.0,

        # Exit parameters (fixed targets work better than reversals)
        profit_target_atr: float = 1.0,  # From analysis: avg win 0.98x ATR
        trailing_stop_atr: float = 1.5,  # Protect profits once in position

        # Indicator parameters
        sma_fast_period: int = 5,
        atr_period: int = 14,

        # Position sizing
        contract_multiplier: float = 2.0,  # MNQ: $2/point
    ):
        self.min_entry_momentum = min_entry_momentum
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        self.profit_target_atr = profit_target_atr
        self.trailing_stop_atr = trailing_stop_atr

        self.sma_fast_period = sma_fast_period
        self.atr_period = atr_period

        self.contract_multiplier = contract_multiplier

        # State
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = None
        self.entry_idx = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        # ATR
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # Moving average
        df['sma_fast'] = df['close'].rolling(self.sma_fast_period).mean()

        # Bar momentum
        df['momentum'] = df['close'] - df['open']

        # ADX calculation
        df['+DM'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['-DM'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )

        # Smooth DM and calculate DI
        df['atr_smooth'] = df['atr'].bfill()
        df['+DI'] = 100 * (df['+DM'].rolling(self.adx_period).mean() / df['atr_smooth'])
        df['-DI'] = 100 * (df['-DM'].rolling(self.adx_period).mean() / df['atr_smooth'])

        # Calculate ADX
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10)
        df['ADX'] = df['DX'].rolling(self.adx_period).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals."""
        df = self.calculate_indicators(df)

        df['signal'] = 0
        df['position_size'] = 0

        position = 0
        entry_price = None
        entry_atr = None
        profit_target = None
        trailing_stop = None

        # Need enough data for indicators
        start_idx = max(self.sma_fast_period, self.atr_period, self.adx_period) + 1

        for i in range(start_idx, len(df)):
            row = df.iloc[i]

            if position == 0:
                # LONG ENTRY: Momentum breakout + close > SMA + ADX > threshold
                long_entry = (
                    row['momentum'] >= self.min_entry_momentum and
                    row['close'] > row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX'])
                )

                # SHORT ENTRY: Opposite conditions
                short_entry = (
                    row['momentum'] <= -self.min_entry_momentum and
                    row['close'] < row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX'])
                )

                if long_entry:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 1
                    position = 1
                    entry_price = row['close']
                    entry_atr = row['atr']
                    profit_target = entry_price + (entry_atr * self.profit_target_atr)
                    trailing_stop = entry_price - (entry_atr * self.trailing_stop_atr)

                elif short_entry:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 1
                    position = -1
                    entry_price = row['close']
                    entry_atr = row['atr']
                    profit_target = entry_price - (entry_atr * self.profit_target_atr)
                    trailing_stop = entry_price + (entry_atr * self.trailing_stop_atr)

            # EXIT LOGIC - FIXED TARGETS + TRAILING STOP
            elif position == 1:  # Long position
                exit_signal = False

                # Profit target hit
                if row['high'] >= profit_target:
                    exit_signal = True

                # Trailing stop hit
                elif row['low'] <= trailing_stop:
                    exit_signal = True

                if exit_signal:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                else:
                    # Update trailing stop (only move up)
                    new_stop = row['close'] - (row['atr'] * self.trailing_stop_atr)
                    trailing_stop = max(trailing_stop, new_stop)
                    df.at[df.index[i], 'position_size'] = 1

            elif position == -1:  # Short position
                exit_signal = False

                # Profit target hit
                if row['low'] <= profit_target:
                    exit_signal = True

                # Trailing stop hit
                elif row['high'] >= trailing_stop:
                    exit_signal = True

                if exit_signal:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                else:
                    # Update trailing stop (only move down)
                    new_stop = row['close'] + (row['atr'] * self.trailing_stop_atr)
                    trailing_stop = min(trailing_stop, new_stop)
                    df.at[df.index[i], 'position_size'] = -1

        return df

    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict:
        """Run backtest and return performance metrics."""
        df = self.generate_signals(df)

        # Calculate returns
        df['strategy_returns'] = 0.0
        position = 0
        entry_price = None

        trades = []
        equity_curve = [initial_capital]
        current_equity = initial_capital

        for i in range(len(df)):
            row = df.iloc[i]

            if position == 0 and row['signal'] != 0:
                # Entry
                position = row['signal']
                entry_price = row['close']
                entry_idx = i

            elif position != 0 and row['signal'] == -position:
                # Exit
                exit_price = row['close']

                if position == 1:
                    pnl_points = exit_price - entry_price
                else:
                    pnl_points = entry_price - exit_price

                pnl_dollars = pnl_points * self.contract_multiplier
                current_equity += pnl_dollars

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_points': pnl_points,
                    'pnl_dollars': pnl_dollars,
                    'bars_held': i - entry_idx,
                })

                position = 0
                entry_price = None

            equity_curve.append(current_equity)

        # Calculate metrics
        if len(trades) == 0:
            return {'error': 'No trades generated'}

        trades_df = pd.DataFrame(trades)

        total_return = (current_equity - initial_capital) / initial_capital
        winners = trades_df[trades_df['pnl_dollars'] > 0]
        losers = trades_df[trades_df['pnl_dollars'] <= 0]

        # Sharpe ratio (annualized)
        returns = trades_df['pnl_dollars'] / initial_capital
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = abs(drawdown.min())

        metrics = {
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': len(winners) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winners['pnl_dollars'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl_dollars'].mean() if len(losers) > 0 else 0,
            'profit_factor': abs(winners['pnl_dollars'].sum() / losers['pnl_dollars'].sum()) if len(losers) > 0 and losers['pnl_dollars'].sum() != 0 else float('inf'),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_bars_held': trades_df['bars_held'].mean(),
            'total_pnl_points': trades_df['pnl_points'].sum(),
            'total_pnl_dollars': trades_df['pnl_dollars'].sum(),
        }

        return metrics, trades_df, df
