"""Backtesting engine, result containers, and parameter sweep optimization.

Provides BacktestEngine for running VectorBT-powered backtests,
BacktestResult dataclass for storing metrics and time-series data,
and ParameterSweep for brute-force grid search optimization.
"""

from simple_futures_backtester.backtest.engine import BacktestEngine, BacktestResult
from simple_futures_backtester.backtest.sweep import ParameterSweep, SweepResult

__all__: list[str] = ["BacktestEngine", "BacktestResult", "ParameterSweep", "SweepResult"]
