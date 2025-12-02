"""Target Trend Strategy - BigBeluga Implementation.

Based on the Target Trend indicator by BigBeluga.
Uses smoothed ATR bands for entry and trailing stops for exit.

Strategy Logic:
- Entry: Close crosses above SMA(high, 10) + ATR_smoothed for longs
         Close crosses below SMA(low, 10) - ATR_smoothed for shorts
- ATR_smoothed: SMA(ATR(200), 200) * 0.8
- Exit: Trailing stop at the entry band level
- Targets: 5x, 10x, 15x ATR (for analysis, not exit)
"""
from typing import Dict
import pandas as pd
import numpy as np


class TargetTrendStrategy:
    """Target Trend strategy with smoothed ATR bands and trailing stops."""

    def __init__(
        self,
        # Entry parameters
        trend_length: int = 10,
        atr_period: int = 200,
        atr_multiplier: float = 0.8,

        # Target parameters (for analysis, not exit)
        target_1_multiplier: float = 5.0,
        target_2_multiplier: float = 10.0,
        target_3_multiplier: float = 15.0,

        # Position sizing
        contract_multiplier: float = 2.0,
    ):
        self.trend_length = trend_length
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        self.target_1_multiplier = target_1_multiplier
        self.target_2_multiplier = target_2_multiplier
        self.target_3_multiplier = target_3_multiplier

        self.contract_multiplier = contract_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        # ATR calculation
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # Smoothed ATR (key to Target Trend)
        df['atr_smoothed'] = df['atr'].rolling(self.atr_period).mean() * self.atr_multiplier

        # Moving average bands
        df['sma_high'] = df['high'].rolling(self.trend_length).mean()
        df['sma_low'] = df['low'].rolling(self.trend_length).mean()

        # Entry bands
        df['upper_band'] = df['sma_high'] + df['atr_smoothed']
        df['lower_band'] = df['sma_low'] - df['atr_smoothed']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals."""
        df = self.calculate_indicators(df)

        df['signal'] = 0
        df['position_size'] = 0
        df['stop_level'] = np.nan

        position = 0
        entry_price = None
        stop_level = None

        start_idx = max(self.trend_length, self.atr_period * 2) + 1

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            if position == 0:
                # LONG ENTRY: Close crosses above upper band
                long_entry = (
                    row['close'] > row['upper_band'] and
                    prev_row['close'] <= prev_row['upper_band'] and
                    not pd.isna(row['upper_band']) and
                    not pd.isna(row['lower_band'])
                )

                # SHORT ENTRY: Close crosses below lower band
                short_entry = (
                    row['close'] < row['lower_band'] and
                    prev_row['close'] >= prev_row['lower_band'] and
                    not pd.isna(row['upper_band']) and
                    not pd.isna(row['lower_band'])
                )

                if long_entry:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 1
                    position = 1
                    entry_price = row['close']
                    stop_level = row['lower_band']  # Stop at lower band for longs
                    df.at[df.index[i], 'stop_level'] = stop_level

                elif short_entry:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 1
                    position = -1
                    entry_price = row['close']
                    stop_level = row['upper_band']  # Stop at upper band for shorts
                    df.at[df.index[i], 'stop_level'] = stop_level

            # EXIT LOGIC - Trailing stop
            elif position == 1:  # Long position
                # Update trailing stop to current lower band
                current_stop = row['lower_band']

                # Exit if price hits stop
                if row['close'] < current_stop or row['low'] <= current_stop:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                    stop_level = None
                else:
                    df.at[df.index[i], 'position_size'] = 1
                    df.at[df.index[i], 'stop_level'] = current_stop

            elif position == -1:  # Short position
                # Update trailing stop to current upper band
                current_stop = row['upper_band']

                # Exit if price hits stop
                if row['close'] > current_stop or row['high'] >= current_stop:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                    stop_level = None
                else:
                    df.at[df.index[i], 'position_size'] = -1
                    df.at[df.index[i], 'stop_level'] = current_stop

        return df

    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict:
        """Run backtest and return performance metrics."""
        df = self.generate_signals(df)

        position = 0
        entry_price = None
        entry_idx = None

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
                entry_idx = None

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
