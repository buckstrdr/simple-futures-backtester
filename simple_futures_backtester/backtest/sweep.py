"""Parameter sweep for brute-force grid search optimization.

Provides ParameterSweep class for systematically evaluating all combinations
of strategy parameters using itertools.product. Results are ranked by Sharpe
ratio to identify optimal parameter configurations.

Supports parallel execution via ProcessPoolExecutor for performance:
- n_jobs=1: Sequential execution (default, for debugging/profiling)
- n_jobs=-1: Use all available CPU cores
- n_jobs>1: Use specified number of workers

Usage:
    >>> from simple_futures_backtester.backtest.sweep import ParameterSweep, SweepResult
    >>> from simple_futures_backtester.config import SweepConfig
    >>> import numpy as np
    >>>
    >>> # Define parameter grid
    >>> sweep_config = SweepConfig(
    ...     strategy="momentum",
    ...     parameters={
    ...         "rsi_period": [10, 14, 20],
    ...         "fast_ema": [5, 9],
    ...         "slow_ema": [21, 50],
    ...     },
    ...     backtest_overrides={"initial_capital": 50000.0},
    ... )
    >>>
    >>> # Run sweep (sequential)
    >>> sweeper = ParameterSweep(n_jobs=1)
    >>> result = sweeper.run(...)
    >>>
    >>> # Run sweep (parallel with all cores)
    >>> sweeper = ParameterSweep(n_jobs=-1)
    >>> result = sweeper.run(
    ...     open_arr=open_prices,
    ...     high_arr=high_prices,
    ...     low_arr=low_prices,
    ...     close_arr=close_prices,
    ...     volume_arr=volume,
    ...     sweep_config=sweep_config,
    ... )
    >>>
    >>> # Access best parameters
    >>> print(f"Best params: {result.best_params}")
    >>> print(f"Best Sharpe: {result.best_sharpe:.3f}")
    >>> print(f"Total combinations tested: {len(result.all_results)}")
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

# Import strategy examples at module level to populate registry in worker processes
# This ensures the registry is available when ProcessPoolExecutor workers start
import simple_futures_backtester.strategy.examples.breakout  # noqa: F401
import simple_futures_backtester.strategy.examples.mean_reversion  # noqa: F401
import simple_futures_backtester.strategy.examples.momentum  # noqa: F401
from simple_futures_backtester.backtest.engine import BacktestEngine, BacktestResult
from simple_futures_backtester.config import BacktestConfig, StrategyConfig, SweepConfig
from simple_futures_backtester.strategy.base import get_strategy

if TYPE_CHECKING:
    pass


@dataclass
class SweepResult:
    """Results from a parameter sweep optimization.

    Contains the best parameter combination found, its Sharpe ratio, and
    complete results for all tested combinations sorted by performance.

    Attributes:
        best_params: Parameter combination with highest Sharpe ratio.
        best_sharpe: The Sharpe ratio of the best parameter combination.
        all_results: List of (params_dict, BacktestResult) tuples sorted
            by Sharpe ratio in descending order (best first).

    Example:
        >>> result = sweeper.run(...)
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best Sharpe: {result.best_sharpe:.3f}")
        >>>
        >>> # Iterate over top 5 results
        >>> for params, backtest_result in result.all_results[:5]:
        ...     print(f"Sharpe: {backtest_result.sharpe_ratio:.3f}, Params: {params}")
    """

    best_params: dict[str, Any]
    best_sharpe: float
    all_results: list[tuple[dict[str, Any], BacktestResult]]


def _run_single_backtest(
    param_combo: tuple[Any, ...],
    param_names: list[str],
    strategy_name: str,
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.int64],
    backtest_config: BacktestConfig,
) -> tuple[dict[str, Any], BacktestResult]:
    """Run a single backtest for one parameter combination.

    This function is pickle-compatible for ProcessPoolExecutor. It is defined
    at module level (not as a method) to ensure it can be serialized and sent
    to worker processes.

    Args:
        param_combo: Tuple of parameter values for this combination.
        param_names: List of parameter names corresponding to combo values.
        strategy_name: Name of the strategy to instantiate from registry.
        open_arr: Opening prices as float64 array.
        high_arr: High prices as float64 array.
        low_arr: Low prices as float64 array.
        close_arr: Closing prices as float64 array.
        volume_arr: Volume as int64 array.
        backtest_config: BacktestConfig with capital, fees, slippage, etc.

    Returns:
        Tuple of (parameter_dict, BacktestResult) for this combination.

    Raises:
        KeyError: If strategy_name is not registered in the strategy registry.
    """
    # Create parameter dictionary
    param_dict = dict(zip(param_names, param_combo, strict=True))

    # Get strategy class from registry (populated via module-level imports)
    StrategyClass = get_strategy(strategy_name)

    # Create strategy config and instance
    strategy_config = StrategyConfig(
        name=strategy_name,
        parameters=param_dict,
    )
    strategy = StrategyClass(strategy_config)

    # Generate signals
    signals = strategy.generate_signals(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        volume_arr,
    )

    # Run backtest with fresh engine instance (stateless, safe for workers)
    engine = BacktestEngine()
    result = engine.run(close_arr, signals, backtest_config)

    return (param_dict, result)


class ParameterSweep:
    """Brute-force grid search optimizer for strategy parameters.

    Systematically tests all combinations of parameter values using
    itertools.product to generate the complete parameter grid. Each
    combination is backtested and results are ranked by Sharpe ratio.

    Supports parallel execution via ProcessPoolExecutor:
    - n_jobs=1: Sequential execution (default, for debugging/profiling)
    - n_jobs=-1: Use all available CPU cores (os.cpu_count())
    - n_jobs>1: Use specified number of workers

    Performance target: 100 parameter combinations in <10 seconds.

    Example:
        >>> # Sequential execution (for debugging)
        >>> sweeper = ParameterSweep(n_jobs=1)
        >>>
        >>> # Parallel execution with all cores
        >>> sweeper = ParameterSweep(n_jobs=-1)
        >>>
        >>> sweep_config = SweepConfig(
        ...     strategy="momentum",
        ...     parameters={"rsi_period": [10, 14], "threshold": [0.5, 0.7]},
        ... )
        >>>
        >>> # With progress tracking
        >>> def progress_callback(current, total):
        ...     print(f"Progress: {current}/{total}")
        >>>
        >>> result = sweeper.run(
        ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
        ...     sweep_config=sweep_config,
        ...     progress_callback=progress_callback,
        ... )
        >>>
        >>> # 2x2 = 4 combinations tested
        >>> assert len(result.all_results) == 4

    Note:
        - Empty parameter grids are handled gracefully (returns empty results)
        - Strategies with no trades produce Sharpe ratio of 0.0
        - NaN Sharpe ratios are treated as 0.0 for sorting purposes
        - Parallel mode creates fresh BacktestEngine instances per worker
    """

    def __init__(self, n_jobs: int = 1) -> None:
        """Initialize ParameterSweep instance.

        Args:
            n_jobs: Number of parallel workers for executing backtests.
                - n_jobs=1: Sequential execution (default). Best for debugging
                  and profiling since it runs in the main process.
                - n_jobs=-1: Use all available CPU cores (os.cpu_count()).
                  Best for maximum throughput on multi-core systems.
                - n_jobs>1: Use exactly this many worker processes.
                  Useful for limiting resource usage.

        Raises:
            ValueError: If n_jobs < 1 and n_jobs != -1.
        """
        if n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        elif n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1 or -1, got {n_jobs}")
        else:
            self.n_jobs = n_jobs

        # Engine instance for sequential mode (n_jobs=1)
        self._engine = BacktestEngine()

    def run(
        self,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
        sweep_config: SweepConfig,
        base_backtest_config: BacktestConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SweepResult:
        """Execute grid search over all parameter combinations.

        Generates all combinations from the parameter grid using itertools.product,
        runs a backtest for each combination, and returns results sorted by
        Sharpe ratio (descending).

        Execution mode depends on n_jobs setting from __init__:
        - n_jobs=1: Sequential execution in main process (for debugging)
        - n_jobs>1 or n_jobs=-1: Parallel execution via ProcessPoolExecutor

        Args:
            open_arr: Opening prices as float64 array.
            high_arr: High prices as float64 array.
            low_arr: Low prices as float64 array.
            close_arr: Closing prices as float64 array.
            volume_arr: Volume as int64 array.
            sweep_config: Configuration specifying strategy name, parameter grid,
                and optional backtest overrides.
            base_backtest_config: Optional base BacktestConfig to use. If None,
                uses default BacktestConfig. Overrides from sweep_config are
                applied on top of this.
            progress_callback: Optional callback function receiving (current, total)
                after each backtest completes. Useful for progress bar integration
                (e.g., Rich progress bars). Called with current number of completed
                backtests and total number of combinations.

        Returns:
            SweepResult containing:
                - best_params: Parameter dict with highest Sharpe ratio
                - best_sharpe: The best Sharpe ratio found
                - all_results: Complete list of (params, BacktestResult) tuples
                    sorted by Sharpe ratio (descending)

        Raises:
            KeyError: If the strategy specified in sweep_config is not registered.
            ValueError: If any backtest fails due to invalid data.
            RuntimeError: If a worker process fails during parallel execution.

        Example:
            >>> sweep_config = SweepConfig(
            ...     strategy="momentum",
            ...     parameters={"rsi_period": [10, 14, 20]},
            ...     backtest_overrides={"fees": 0.0002},
            ... )
            >>>
            >>> # With progress callback for Rich progress bar
            >>> def on_progress(current, total):
            ...     print(f"Completed {current}/{total}")
            >>>
            >>> result = sweeper.run(
            ...     open_arr, high_arr, low_arr, close_arr, volume_arr,
            ...     sweep_config=sweep_config,
            ...     progress_callback=on_progress,
            ... )
            >>> print(f"Tested {len(result.all_results)} combinations")
        """
        # Handle empty parameter grid
        if not sweep_config.parameters:
            return SweepResult(
                best_params={},
                best_sharpe=0.0,
                all_results=[],
            )

        # Build backtest configuration with overrides
        backtest_config = self._build_backtest_config(
            base_backtest_config,
            sweep_config.backtest_overrides,
        )

        # Generate all parameter combinations
        param_names = list(sweep_config.parameters.keys())
        param_values = list(sweep_config.parameters.values())
        all_combos = list(itertools.product(*param_values))
        total_combos = len(all_combos)

        all_results: list[tuple[dict[str, Any], BacktestResult]] = []

        if self.n_jobs == 1:
            # Sequential execution path (for debugging/profiling)
            all_results = self._run_sequential(
                all_combos,
                param_names,
                sweep_config.strategy,
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                volume_arr,
                backtest_config,
                progress_callback,
                total_combos,
            )
        else:
            # Parallel execution path via ProcessPoolExecutor
            all_results = self._run_parallel(
                all_combos,
                param_names,
                sweep_config.strategy,
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                volume_arr,
                backtest_config,
                progress_callback,
                total_combos,
            )

        # Sort by Sharpe ratio (descending), treating NaN as 0
        all_results_sorted = sorted(
            all_results,
            key=lambda x: self._safe_sharpe(x[1].sharpe_ratio),
            reverse=True,
        )

        # Extract best parameters and Sharpe
        if all_results_sorted:
            best_params, best_result = all_results_sorted[0]
            best_sharpe = self._safe_sharpe(best_result.sharpe_ratio)
        else:
            best_params = {}
            best_sharpe = 0.0

        return SweepResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            all_results=all_results_sorted,
        )

    def _run_sequential(
        self,
        all_combos: list[tuple[Any, ...]],
        param_names: list[str],
        strategy_name: str,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
        backtest_config: BacktestConfig,
        progress_callback: Callable[[int, int], None] | None,
        total_combos: int,
    ) -> list[tuple[dict[str, Any], BacktestResult]]:
        """Run backtests sequentially in the main process.

        Used when n_jobs=1 for debugging and profiling purposes.

        Args:
            all_combos: List of parameter value tuples to test.
            param_names: List of parameter names.
            strategy_name: Name of strategy in registry.
            open_arr: Opening prices.
            high_arr: High prices.
            low_arr: Low prices.
            close_arr: Closing prices.
            volume_arr: Volume data.
            backtest_config: Backtest configuration.
            progress_callback: Optional progress callback.
            total_combos: Total number of combinations.

        Returns:
            List of (param_dict, BacktestResult) tuples.
        """
        # Get strategy class from registry once
        StrategyClass = get_strategy(strategy_name)

        results: list[tuple[dict[str, Any], BacktestResult]] = []

        for idx, combo_values in enumerate(all_combos):
            # Build parameter dictionary for this combination
            param_dict = dict(zip(param_names, combo_values, strict=True))

            # Create strategy config and instance
            strategy_config = StrategyConfig(
                name=strategy_name,
                parameters=param_dict,
            )
            strategy = StrategyClass(strategy_config)

            # Generate signals
            signals = strategy.generate_signals(
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                volume_arr,
            )

            # Run backtest using instance engine
            result = self._engine.run(close_arr, signals, backtest_config)

            # Store result
            results.append((param_dict, result))

            # Report progress
            if progress_callback is not None:
                progress_callback(idx + 1, total_combos)

        return results

    def _run_parallel(
        self,
        all_combos: list[tuple[Any, ...]],
        param_names: list[str],
        strategy_name: str,
        open_arr: NDArray[np.float64],
        high_arr: NDArray[np.float64],
        low_arr: NDArray[np.float64],
        close_arr: NDArray[np.float64],
        volume_arr: NDArray[np.int64],
        backtest_config: BacktestConfig,
        progress_callback: Callable[[int, int], None] | None,
        total_combos: int,
    ) -> list[tuple[dict[str, Any], BacktestResult]]:
        """Run backtests in parallel using ProcessPoolExecutor.

        Used when n_jobs > 1 for improved throughput on multi-core systems.

        Args:
            all_combos: List of parameter value tuples to test.
            param_names: List of parameter names.
            strategy_name: Name of strategy in registry.
            open_arr: Opening prices.
            high_arr: High prices.
            low_arr: Low prices.
            close_arr: Closing prices.
            volume_arr: Volume data.
            backtest_config: Backtest configuration.
            progress_callback: Optional progress callback.
            total_combos: Total number of combinations.

        Returns:
            List of (param_dict, BacktestResult) tuples.

        Raises:
            RuntimeError: If a worker process fails.
        """
        results: list[tuple[dict[str, Any], BacktestResult]] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all jobs to the pool
            futures = {
                executor.submit(
                    _run_single_backtest,
                    combo,
                    param_names,
                    strategy_name,
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    volume_arr,
                    backtest_config,
                ): combo
                for combo in all_combos
            }

            # Collect results as they complete
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Get the combo that failed for error message
                    failed_combo = futures[future]
                    raise RuntimeError(
                        f"Worker process failed for parameter combination "
                        f"{dict(zip(param_names, failed_combo, strict=True))}"
                    ) from e

                if progress_callback is not None:
                    progress_callback(completed, total_combos)

        return results

    def _build_backtest_config(
        self,
        base_config: BacktestConfig | None,
        overrides: dict[str, Any],
    ) -> BacktestConfig:
        """Build BacktestConfig with overrides applied.

        Args:
            base_config: Base configuration to start from. If None, uses defaults.
            overrides: Dictionary of field overrides to apply.

        Returns:
            BacktestConfig with overrides applied.
        """
        if base_config is None:
            base_config = BacktestConfig()

        if not overrides:
            return base_config

        return BacktestConfig(
            initial_capital=float(
                overrides.get("initial_capital", base_config.initial_capital)
            ),
            fees=float(overrides.get("fees", base_config.fees)),
            slippage=float(overrides.get("slippage", base_config.slippage)),
            size=int(overrides.get("size", base_config.size)),
            size_type=str(overrides.get("size_type", base_config.size_type)),
            freq=str(overrides.get("freq", base_config.freq)),
        )

    @staticmethod
    def _safe_sharpe(value: float) -> float:
        """Convert Sharpe ratio to safe float for sorting.

        Handles NaN and infinity values by replacing them with 0.0.

        Args:
            value: Sharpe ratio value.

        Returns:
            Safe float value for comparison.
        """
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value


__all__: list[str] = ["ParameterSweep", "SweepResult"]
