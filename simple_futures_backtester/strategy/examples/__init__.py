"""Example strategy implementations.

Provides reference implementations demonstrating the strategy framework:
- MomentumStrategy: RSI + EMA momentum-based entries
- MeanReversionStrategy: Bollinger Bands mean reversion
- BreakoutStrategy: EMA crossover breakout system
- MNQVortexScalper: Vortex Indicator + HMA scalping strategy
"""

from simple_futures_backtester.strategy.examples.breakout import BreakoutStrategy
from simple_futures_backtester.strategy.examples.mean_reversion import MeanReversionStrategy
from simple_futures_backtester.strategy.examples.mnq_chop_adx import MNQChopADXScalper
from simple_futures_backtester.strategy.examples.mnq_vortex import MNQVortexScalper
from simple_futures_backtester.strategy.examples.momentum import MomentumStrategy

__all__: list[str] = ["BreakoutStrategy", "MeanReversionStrategy", "MomentumStrategy", "MNQVortexScalper", "MNQChopADXScalper"]
