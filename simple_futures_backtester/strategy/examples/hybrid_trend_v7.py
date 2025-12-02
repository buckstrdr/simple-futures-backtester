"""Hybrid Trend V7 - Combining Best of Big Trend V6 + Target Trend.

ENTRY:
- Target Trend's smoothed ATR bands (SMA(high/low, 10) ± SMA(ATR(200), 200) * 0.8)
- Big Trend's momentum filter (8+ points)
- Big Trend's ADX filter (> 30)

EXIT:
- Big Trend's decoded scalp exit (fast loss cutting)
- OR Target Trend's opposite band crossover (trend reversal)

Hypothesis: Target Trend's smoother bands + Big Trend's filters + Big Trend's quick exits
= Better entry selectivity + Better risk management
"""
from typing import Dict
import pandas as pd
import numpy as np


class HybridTrendV7Strategy:
    """Hybrid strategy combining Target Trend entry bands + Big Trend filters + scalp exit."""

    def __init__(
        self,
        # Target Trend entry parameters
        trend_length: int = 10,
        atr_long_period: int = 200,  # Target Trend's long ATR
        atr_multiplier: float = 0.8,

        # Big Trend filters
        min_entry_momentum: float = 8.0,
        adx_threshold: float = 30.0,

        # Big Trend scalp exit parameters
        scalp_exit_momentum_long: float = -5.0,
        scalp_exit_close_position_long: float = 0.35,
        scalp_exit_momentum_short: float = 5.0,
        scalp_exit_close_position_short: float = 0.65,

        # Technical parameters
        adx_period: int = 14,
        contract_multiplier: float = 2.0,
    ):
        # Entry parameters
        self.trend_length = trend_length
        self.atr_long_period = atr_long_period
        self.atr_multiplier = atr_multiplier

        # Filters
        self.min_entry_momentum = min_entry_momentum
        self.adx_threshold = adx_threshold

        # Exit parameters
        self.scalp_exit_momentum_long = scalp_exit_momentum_long
        self.scalp_exit_close_position_long = scalp_exit_close_position_long
        self.scalp_exit_momentum_short = scalp_exit_momentum_short
        self.scalp_exit_close_position_short = scalp_exit_close_position_short

        self.adx_period = adx_period
        self.contract_multiplier = contract_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        # ATR (long period for Target Trend smoothing)
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        df['atr_raw'] = df['tr'].rolling(self.atr_long_period).mean()

        # Smoothed ATR (Target Trend style)
        df['atr_smoothed'] = df['atr_raw'].rolling(self.atr_long_period).mean() * self.atr_multiplier

        # Target Trend bands
        df['sma_high'] = df['high'].rolling(self.trend_length).mean()
        df['sma_low'] = df['low'].rolling(self.trend_length).mean()
        df['upper_band'] = df['sma_high'] + df['atr_smoothed']
        df['lower_band'] = df['sma_low'] - df['atr_smoothed']

        # Big Trend momentum
        df['momentum'] = df['close'] - df['open']

        # Close position in bar (for scalp exit)
        df['bar_range'] = df['high'] - df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['bar_range'] + 1e-10)

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

        df['atr_smooth'] = df['atr_raw'].bfill()
        df['+DI'] = 100 * (df['+DM'].rolling(self.adx_period).mean() / df['atr_smooth'])
        df['-DI'] = 100 * (df['-DM'].rolling(self.adx_period).mean() / df['atr_smooth'])
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

        start_idx = max(self.trend_length, self.atr_long_period * 2, self.adx_period) + 1

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            if position == 0:
                # LONG ENTRY: Target Trend crossover + Big Trend filters
                long_entry = (
                    row['close'] > row['upper_band'] and
                    prev_row['close'] <= prev_row['upper_band'] and
                    row['momentum'] >= self.min_entry_momentum and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX']) and
                    not pd.isna(row['upper_band'])
                )

                # SHORT ENTRY: Target Trend crossunder + Big Trend filters
                short_entry = (
                    row['close'] < row['lower_band'] and
                    prev_row['close'] >= prev_row['lower_band'] and
                    row['momentum'] <= -self.min_entry_momentum and
                    row['ADX'] > self.adx_threshold and
                    not pd.isna(row['ADX']) and
                    not pd.isna(row['lower_band'])
                )

                if long_entry:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 1
                    position = 1
                    entry_price = row['close']

                elif short_entry:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 1
                    position = -1
                    entry_price = row['close']

            # EXIT LOGIC - Dual exits
            elif position == 1:  # Long position
                # Exit 1: Big Trend scalp exit (fast loss cutting)
                scalp_exit = (
                    row['momentum'] < self.scalp_exit_momentum_long and
                    row['close_position'] < self.scalp_exit_close_position_long
                )

                # Exit 2: Target Trend opposite crossover (trend reversal)
                trend_exit = (
                    row['close'] < row['lower_band'] and
                    prev_row['close'] >= prev_row['lower_band']
                )

                if scalp_exit or trend_exit:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                else:
                    df.at[df.index[i], 'position_size'] = 1

            elif position == -1:  # Short position
                # Exit 1: Big Trend scalp exit (fast loss cutting)
                scalp_exit = (
                    row['momentum'] > self.scalp_exit_momentum_short and
                    row['close_position'] > self.scalp_exit_close_position_short
                )

                # Exit 2: Target Trend opposite crossover (trend reversal)
                trend_exit = (
                    row['close'] > row['upper_band'] and
                    prev_row['close'] <= prev_row['upper_band']
                )

                if scalp_exit or trend_exit:
                    df.at[df.index[i], 'signal'] = 1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                else:
                    df.at[df.index[i], 'position_size'] = -1

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
