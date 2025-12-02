"""Simple Futures Backtester - High-performance vectorized backtesting framework.

A high-performance, vectorized futures backtesting framework built on VectorBT.
Provides JIT-compiled bar factories, trailing stop implementations, and
comprehensive portfolio analytics for futures trading research.
"""

import os

__version__ = "0.1.0"
__author__ = "Simple Futures Backtester Team"

# Skip CLI import during testing to avoid VectorBT/Plotly incompatibility
_TESTING = os.environ.get("SFB_TESTING", "0") == "1"

if not _TESTING:
    from simple_futures_backtester.cli import app

from simple_futures_backtester.config import (
    BacktestConfig,
    LoadedConfig,
    StrategyConfig,
    SweepConfig,
    compute_config_hash,
    load_config,
    load_strategy_config,
    load_sweep_config,
)

__all__ = [
    "__version__",
    "BacktestConfig",
    "LoadedConfig",
    "StrategyConfig",
    "SweepConfig",
    "compute_config_hash",
    "load_config",
    "load_strategy_config",
    "load_sweep_config",
]

if not _TESTING:
    __all__.append("app")
