"""VectorBT extensions for futures backtesting.

Provides custom extensions to VectorBT:
- Delayed and ATR-based trailing stops (JIT-compiled)
- FuturesPortfolio wrapper with point value application
- Custom indicator factories
"""

from simple_futures_backtester.extensions.futures_portfolio import (
    FuturesPortfolio,
    PortfolioAnalytics,
)
from simple_futures_backtester.extensions.trailing_stops import (
    atr_trailing_stop_nb,
    delayed_trailing_stop_nb,
    generate_trailing_exits,
)

__all__: list[str] = [
    "FuturesPortfolio",
    "PortfolioAnalytics",
    "delayed_trailing_stop_nb",
    "atr_trailing_stop_nb",
    "generate_trailing_exits",
]
