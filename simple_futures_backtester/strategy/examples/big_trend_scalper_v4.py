"""Big Trend Double Check Strategy - Version 4 (Tuned Filters).

V3 had 690 trades (too few), need ~2225. Adjustments:
- ADX 22 instead of 25 (slightly relaxed)
- Volume 1.1x instead of 1.2x (relaxed)
- ATR threshold 12 instead of 15 (relaxed)
- Trailing stop 1.2x instead of 1.5x (tighter to reduce avg loss)

Strategy Logic:
- Entry: Momentum breakout + Close > SMA(5) + ADX > 22 + Volume > 1.1x avg + ATR > 12
- Exit: Fixed profit target (1.0x ATR) OR trailing stop (1.2x ATR)
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np


class BigTrendScalperV4Strategy:
    """Big Trend scalping strategy with tuned filters."""

    def __init__(
        self,
        # Entry parameters
        min_entry_momentum: float = 8.0,
        adx_period: int = 14,
        adx_threshold: float = 22.0,  # TUNED: 22 (was 25 in V3)

        # Volume filter
        volume_ma_period: int = 20,
        min_volume_ratio: float = 1.1,  # TUNED: 1.1x (was 1.2x in V3)

        # ATR filter
        min_atr_threshold: float = 12.0,  # TUNED: 12 (was 15 in V3)

        # Exit parameters
        profit_target_atr: float = 1.0,
        trailing_stop_atr: float = 1.2,  # TUNED: 1.2x (was 1.5x in V3)

        # Indicator parameters
        sma_fast_period: int = 5,
        atr_period: int = 14,

        # Position sizing
        contract_multiplier: float = 2.0,
    ):
        self.min_entry_momentum = min_entry_momentum
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio
        self.min_atr_threshold = min_atr_threshold

        self.profit_target_atr = profit_target_atr
        self.trailing_stop_atr = trailing_stop_atr

        self.sma_fast_period = sma_fast_period
        self.atr_period = atr_period

        self.contract_multiplier = contract_multiplier

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

        # Volume filter
        df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

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
        start_idx = max(
            self.sma_fast_period,
            self.atr_period,
            self.adx_period,
            self.volume_ma_period
        ) + 1

        for i in range(start_idx, len(df)):
            row = df.iloc[i]

            if position == 0:
                # LONG ENTRY
                long_entry = (
                    row['momentum'] >= self.min_entry_momentum and
                    row['close'] > row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX']) and
                    row['volume_ratio'] > self.min_volume_ratio and
                    row['atr'] > self.min_atr_threshold
                )

                # SHORT ENTRY
                short_entry = (
                    row['momentum'] <= -self.min_entry_momentum and
                    row['close'] < row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX']) and
                    row['volume_ratio'] > self.min_volume_ratio and
                    row['atr'] > self.min_atr_threshold
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

            # EXIT LOGIC
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
