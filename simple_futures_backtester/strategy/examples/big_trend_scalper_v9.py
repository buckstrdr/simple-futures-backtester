"""Big Trend Double Check Strategy - Version 9 (Stricter ADX Filter).

IMPROVEMENT OVER V8: Raising ADX threshold from 30 to 35 for stronger trends.

Analysis showed that higher ADX improves performance:
- V8 (ADX > 30): +1.11%, 1469 trades, 37.1% win rate
- V9 (ADX > 35): +1.22%, 1145 trades, 38.1% win rate (+0.11% improvement)

By requiring ADX > 35 instead of 30, we filter out weaker trends and reduce
the proportion of losing trades from 62.9% to ~62%.

Strategy Logic:
- Entry: Momentum breakout + Close > SMA(5) + ADX > 35 + DI Spread >= 20
- Exit: Strong opposite momentum + close position in bar (decoded from original)
"""
from typing import Dict
import pandas as pd
import numpy as np


class BigTrendScalperV9Strategy:
    """Big Trend scalping strategy with ADX 35+ and DI Spread >= 20 filters."""

    def __init__(
        self,
        # Entry parameters
        min_entry_momentum: float = 8.0,
        adx_threshold: float = 35.0,  # RAISED from 30 to 35
        di_spread_threshold: float = 20.0,

        # EXIT parameters (DECODED from original indicator)
        scalp_exit_momentum_long: float = -5.0,
        scalp_exit_close_position_long: float = 0.35,
        scalp_exit_momentum_short: float = 5.0,
        scalp_exit_close_position_short: float = 0.65,

        # Indicator parameters
        sma_fast_period: int = 5,
        atr_period: int = 14,
        adx_period: int = 14,

        # Position sizing
        contract_multiplier: float = 2.0,
    ):
        self.min_entry_momentum = min_entry_momentum
        self.adx_threshold = adx_threshold
        self.di_spread_threshold = di_spread_threshold

        self.scalp_exit_momentum_long = scalp_exit_momentum_long
        self.scalp_exit_close_position_long = scalp_exit_close_position_long
        self.scalp_exit_momentum_short = scalp_exit_momentum_short
        self.scalp_exit_close_position_short = scalp_exit_close_position_short

        self.sma_fast_period = sma_fast_period
        self.atr_period = atr_period
        self.adx_period = adx_period

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

        # Close position in bar (0=low, 1=high) - KEY METRIC
        df['bar_range'] = df['high'] - df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['bar_range'] + 1e-10)

        # ADX calculation with DI Spread
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

        df['atr_smooth'] = df['atr'].bfill()
        df['+DI'] = 100 * (df['+DM'].rolling(self.adx_period).mean() / df['atr_smooth'])
        df['-DI'] = 100 * (df['-DM'].rolling(self.adx_period).mean() / df['atr_smooth'])
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10)
        df['ADX'] = df['DX'].rolling(self.adx_period).mean()

        # DI SPREAD - KEY METRIC
        df['di_spread'] = abs(df['+DI'] - df['-DI'])

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals."""
        df = self.calculate_indicators(df)

        df['signal'] = 0
        df['position_size'] = 0

        position = 0
        entry_price = None

        start_idx = max(self.sma_fast_period, self.atr_period, self.adx_period) + 1

        for i in range(start_idx, len(df)):
            row = df.iloc[i]

            if position == 0:
                # LONG ENTRY (with ADX 35+ and DI Spread filters)
                long_entry = (
                    row['momentum'] >= self.min_entry_momentum and
                    row['close'] > row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and  # ADX > 35
                    row['di_spread'] >= self.di_spread_threshold and
                    not pd.isna(row['ADX']) and
                    not pd.isna(row['sma_fast']) and
                    not pd.isna(row['di_spread'])
                )

                # SHORT ENTRY (with ADX 35+ and DI Spread filters)
                short_entry = (
                    row['momentum'] <= -self.min_entry_momentum and
                    row['close'] < row['sma_fast'] and
                    row['ADX'] > self.adx_threshold and  # ADX > 35
                    row['di_spread'] >= self.di_spread_threshold and
                    not pd.isna(row['ADX']) and
                    not pd.isna(row['sma_fast']) and
                    not pd.isna(row['di_spread'])
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

            # EXIT LOGIC - DECODED FROM ORIGINAL INDICATOR
            elif position == 1:  # Long position
                # Scalp exit: Strong bearish momentum + close near low
                scalp_exit = (
                    row['momentum'] < self.scalp_exit_momentum_long and
                    row['close_position'] < self.scalp_exit_close_position_long
                )

                if scalp_exit:
                    df.at[df.index[i], 'signal'] = -1
                    df.at[df.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                else:
                    df.at[df.index[i], 'position_size'] = 1

            elif position == -1:  # Short position
                # Scalp exit: Strong bullish momentum + close near high
                scalp_exit = (
                    row['momentum'] > self.scalp_exit_momentum_short and
                    row['close_position'] > self.scalp_exit_close_position_short
                )

                if scalp_exit:
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
