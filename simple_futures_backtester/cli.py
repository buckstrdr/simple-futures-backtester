"""Typer CLI entry point for Simple Futures Backtester.

Provides commands for backtesting, parameter sweeps, bar generation,
benchmarking, and exporting results.
"""

import csv
import itertools
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

# Import and register strategies at module load time
from simple_futures_backtester.strategy.base import get_strategy, list_strategies, register_strategy
from simple_futures_backtester.strategy.examples import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)

# Register all example strategies
register_strategy("momentum", MomentumStrategy)
register_strategy("mean_reversion", MeanReversionStrategy)
register_strategy("breakout", BreakoutStrategy)

app = typer.Typer(
    name="sfb",
    help="Simple Futures Backtester - High-performance vectorized backtesting framework.",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    from simple_futures_backtester import __version__

    console.print(f"Simple Futures Backtester v{__version__}")


@app.command()
def backtest(
    data: str = typer.Option(..., "--data", "-d", help="Path to OHLCV data file (CSV/Parquet)"),
    strategy: str = typer.Option(
        ..., "--strategy", "-s", help="Strategy name (e.g., momentum, breakout)"
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to strategy config YAML"),
    capital: float | None = typer.Option(None, "--capital", help="Override initial capital"),
    fees: float | None = typer.Option(None, "--fees", help="Override transaction fees"),
    slippage: float | None = typer.Option(None, "--slippage", help="Override slippage"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for results"),
) -> None:
    """Run a single backtest with a strategy configuration."""
    from simple_futures_backtester.backtest.engine import BacktestEngine
    from simple_futures_backtester.config import (
        BacktestConfig,
        StrategyConfig,
        load_strategy_config,
    )
    from simple_futures_backtester.data.loader import DataLoadError, load_csv, load_parquet
    from simple_futures_backtester.output.exports import ResultsExporter
    from simple_futures_backtester.output.reports import ReportGenerator

    try:
        # 1. Load and validate data
        console.print("[cyan]Loading data...[/cyan]")
        data_path = Path(data)

        if not data_path.exists():
            console.print(f"[red]Error: File not found - {data}[/red]")
            raise typer.Exit(code=1)

        if data_path.suffix.lower() in [".parquet", ".pq"]:
            df = load_parquet(data_path)
        else:
            df = load_csv(data_path)
        console.print(f"[green]Loaded {len(df)} bars[/green]")

        # 2. Load config or use defaults
        if config:
            config_path = Path(config)
            if not config_path.exists():
                console.print(f"[red]Error: Config file not found - {config}[/red]")
                raise typer.Exit(code=1)
            strategy_config, backtest_config, _ = load_strategy_config(config)
            console.print(f"[green]Loaded config: {config}[/green]")
        else:
            strategy_config = StrategyConfig(name=strategy, parameters={})
            backtest_config = BacktestConfig()

        # 3. Apply CLI overrides
        if capital is not None:
            backtest_config = BacktestConfig(
                initial_capital=capital,
                fees=backtest_config.fees,
                slippage=backtest_config.slippage,
                size=backtest_config.size,
                size_type=backtest_config.size_type,
                freq=backtest_config.freq,
            )
        if fees is not None:
            backtest_config = BacktestConfig(
                initial_capital=backtest_config.initial_capital,
                fees=fees,
                slippage=backtest_config.slippage,
                size=backtest_config.size,
                size_type=backtest_config.size_type,
                freq=backtest_config.freq,
            )
        if slippage is not None:
            backtest_config = BacktestConfig(
                initial_capital=backtest_config.initial_capital,
                fees=backtest_config.fees,
                slippage=slippage,
                size=backtest_config.size,
                size_type=backtest_config.size_type,
                freq=backtest_config.freq,
            )

        # 4. Instantiate strategy
        # Use strategy from config if config was loaded, otherwise use CLI arg
        strategy_name = strategy_config.name if config else strategy
        console.print(f"[cyan]Initializing strategy: {strategy_name}[/cyan]")

        try:
            StrategyClass = get_strategy(strategy_name)
        except KeyError:
            console.print(f"[red]Error: Unknown strategy '{strategy_name}'[/red]")
            available = list_strategies()
            if available:
                console.print(f"Available strategies: {', '.join(available)}")
            else:
                console.print("No strategies registered.")
            raise typer.Exit(code=1) from None

        strategy_instance = StrategyClass(strategy_config)

        # 5. Generate signals
        console.print("[cyan]Generating signals...[/cyan]")
        signals = strategy_instance.generate_signals(
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
        )
        console.print(f"[green]Generated {len(signals)} signals[/green]")

        # 6. Run backtest with progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)
            engine = BacktestEngine()
            result = engine.run(df["close"].values, signals, backtest_config)
            progress.update(task, completed=True, description="Backtest complete")
        console.print("[green]Backtest complete[/green]")

        # 7. Display results
        console.print("\n")
        report = ReportGenerator.generate_text_report(result)
        console.print(report)

        # 8. Export if requested
        if output:
            console.print(f"\n[cyan]Exporting results to {output}/[/cyan]")
            ResultsExporter.export_all(
                result,
                output_dir=output,
                strategy_name=strategy_name,
                close_prices=df["close"].values,
            )
            console.print(f"[green]Results exported to {output}/[/green]")

    except typer.Exit:
        # Re-raise typer.Exit to propagate exit codes properly
        raise
    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(code=1) from None
    except DataLoadError as e:
        console.print(f"[red]Data validation error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except KeyError as e:
        console.print(f"[red]Strategy error: {e}[/red]")
        available = list_strategies()
        if available:
            console.print(f"Available strategies: {', '.join(available)}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None


@app.command()
def sweep(
    data: str = typer.Option(..., "--data", "-d", help="Path to OHLCV data file (CSV/Parquet)"),
    strategy: str = typer.Option(
        ..., "--strategy", "-s", help="Strategy name (e.g., momentum, breakout)"
    ),
    sweep_config: str = typer.Option(
        ..., "--sweep-config", "-c", help="Path to sweep configuration YAML"
    ),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Number of parallel workers (default: 1)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for results"),
) -> None:
    """Execute parameter grid search optimization.

    Systematically tests all combinations of strategy parameters and ranks
    results by Sharpe ratio. Displays top 10 results and optionally exports
    full results to CSV.

    Example:
        sfb sweep --data es.csv --strategy momentum --sweep-config sweep.yaml
    """
    from simple_futures_backtester.backtest.sweep import ParameterSweep
    from simple_futures_backtester.config import load_sweep_config
    from simple_futures_backtester.data.loader import DataLoadError, load_csv, load_parquet

    try:
        # 1. Load and validate data
        console.print("[cyan]Loading data...[/cyan]")
        data_path = Path(data)

        if not data_path.exists():
            console.print(f"[red]Error: File not found - {data}[/red]")
            raise typer.Exit(code=1)

        if data_path.suffix.lower() in [".parquet", ".pq"]:
            df = load_parquet(data_path)
        else:
            df = load_csv(data_path)
        console.print(f"[green]Loaded {len(df)} bars[/green]")

        # 2. Load sweep configuration
        console.print("[cyan]Loading sweep configuration...[/cyan]")
        sweep_config_path = Path(sweep_config)

        if not sweep_config_path.exists():
            console.print(f"[red]Error: Sweep config file not found - {sweep_config}[/red]")
            raise typer.Exit(code=1)

        loaded_sweep_config, backtest_config, config_hash = load_sweep_config(sweep_config_path)
        console.print(f"[green]Loaded config: {sweep_config}[/green]")

        # 3. Validate strategy exists
        strategy_name = strategy if strategy else loaded_sweep_config.strategy
        console.print(f"[cyan]Validating strategy: {strategy_name}[/cyan]")

        try:
            get_strategy(strategy_name)
        except KeyError:
            console.print(f"[red]Error: Unknown strategy '{strategy_name}'[/red]")
            available = list_strategies()
            if available:
                console.print(f"Available strategies: {', '.join(available)}")
            raise typer.Exit(code=1) from None

        # Override strategy in sweep config if CLI arg provided
        if strategy and strategy != loaded_sweep_config.strategy:
            from simple_futures_backtester.config import SweepConfig

            loaded_sweep_config = SweepConfig(
                strategy=strategy,
                parameters=loaded_sweep_config.parameters,
                backtest_overrides=loaded_sweep_config.backtest_overrides,
            )

        # 4. Calculate total combinations
        param_values = list(loaded_sweep_config.parameters.values())
        total_combos = len(list(itertools.product(*param_values))) if param_values else 0

        if total_combos == 0:
            console.print("[yellow]No parameter combinations to test[/yellow]")
            raise typer.Exit(code=0)

        console.print(
            f"[cyan]Testing {total_combos} parameter combinations "
            f"with {n_jobs} worker(s)...[/cyan]"
        )

        # 5. Initialize sweeper and run with progress bar
        sweeper = ParameterSweep(n_jobs=n_jobs)

        # Extract OHLCV arrays
        open_arr = df["open"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        close_arr = df["close"].values
        volume_arr = df["volume"].values

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Testing combinations", total=total_combos)

            def update_progress(current: int, _total: int) -> None:
                progress.update(task, completed=current)

            result = sweeper.run(
                open_arr=open_arr,
                high_arr=high_arr,
                low_arr=low_arr,
                close_arr=close_arr,
                volume_arr=volume_arr,
                sweep_config=loaded_sweep_config,
                base_backtest_config=backtest_config,
                progress_callback=update_progress,
            )

        console.print("[green]Sweep complete[/green]")

        # 6. Display results table (top 10)
        console.print("\n")
        table = Table(title="Top 10 Parameter Combinations by Sharpe Ratio")
        table.add_column("Rank", style="cyan", no_wrap=True, justify="right")
        table.add_column("Sharpe", justify="right", style="green")
        table.add_column("Return", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Parameters", style="dim")

        for idx, (params, backtest_result) in enumerate(result.all_results[:10], start=1):
            table.add_row(
                str(idx),
                f"{backtest_result.sharpe_ratio:.3f}",
                f"{backtest_result.total_return:.2%}",
                f"{backtest_result.win_rate:.2%}",
                str(backtest_result.n_trades),
                str(params),
            )

        console.print(table)

        # 7. Show best result summary
        if result.all_results:
            console.print(f"\n[green]Best parameters:[/green] {result.best_params}")
            console.print(f"[green]Best Sharpe ratio:[/green] {result.best_sharpe:.3f}")

        # 8. Export to CSV if output specified
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            csv_path = output_path / "all_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rank",
                        "sharpe_ratio",
                        "total_return",
                        "win_rate",
                        "n_trades",
                        "parameters",
                    ]
                )

                for idx, (params, br) in enumerate(result.all_results, start=1):
                    writer.writerow(
                        [
                            idx,
                            f"{br.sharpe_ratio:.6f}",
                            f"{br.total_return:.6f}",
                            f"{br.win_rate:.6f}",
                            br.n_trades,
                            str(params),
                        ]
                    )

            console.print(
                f"\n[green]Exported {len(result.all_results)} results " f"to {csv_path}[/green]"
            )

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Sweep interrupted by user[/yellow]")
        raise typer.Exit(code=1) from None
    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(code=1) from None
    except DataLoadError as e:
        console.print(f"[red]Data validation error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except KeyError as e:
        console.print(f"[red]Strategy error: {e}[/red]")
        available = list_strategies()
        if available:
            console.print(f"Available strategies: {', '.join(available)}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None


@app.command(name="generate-bars")
def generate_bars(
    bar_type: str = typer.Option(
        ...,
        "--bar-type",
        "-t",
        help="Bar type: renko, range, tick, volume, dollar, tick_imbalance, volume_imbalance",
    ),
    data: str = typer.Option(..., "--data", "-d", help="Path to OHLCV data file (CSV/Parquet)"),
    param: str = typer.Option(
        ...,
        "--param",
        "-p",
        help=(
            "Bar-specific parameter value. Interpretation depends on --bar-type: "
            "renko=brick_size (float), range=range_size (float), tick=tick_threshold (int), "
            "volume=volume_threshold (int), dollar=dollar_threshold (float), "
            "tick_imbalance=imbalance_threshold (int), volume_imbalance=imbalance_threshold (int)"
        ),
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output CSV file path"),
) -> None:
    """Generate alternative bar types from OHLCV data.

    Transforms standard time-based OHLCV bars into alternative bar types that filter
    market noise and normalize trading activity.

    Example:
        sfb generate-bars --data es.csv --bar-type renko --param 10
    """
    import pandas as pd

    from simple_futures_backtester.bars import get_bar_generator, list_bar_types
    from simple_futures_backtester.data.loader import DataLoadError, load_csv, load_parquet

    # Mapping: bar_type -> (param_name, param_type)
    BAR_PARAM_MAP: dict[str, tuple[str, type]] = {
        "renko": ("brick_size", float),
        "range": ("range_size", float),
        "tick": ("tick_threshold", int),
        "volume": ("volume_threshold", int),
        "dollar": ("dollar_threshold", float),
        "tick_imbalance": ("imbalance_threshold", int),
        "volume_imbalance": ("imbalance_threshold", int),
    }

    try:
        # 1. Validate bar type
        available_types = list_bar_types()
        if bar_type not in available_types:
            console.print(f"[red]Error: Unknown bar type '{bar_type}'[/red]")
            console.print(f"Available types: {', '.join(available_types)}")
            raise typer.Exit(code=1)

        # 2. Get parameter mapping
        if bar_type not in BAR_PARAM_MAP:
            console.print(f"[red]Error: Bar type '{bar_type}' has no parameter mapping[/red]")
            raise typer.Exit(code=1)

        param_name, param_type = BAR_PARAM_MAP[bar_type]

        # 3. Convert param string to correct type
        try:
            param_value = param_type(param)
        except (ValueError, TypeError) as e:
            console.print(
                f"[red]Error: Invalid {param_name} value '{param}' for {bar_type} bars: {e}[/red]"
            )
            if param_type is int:
                console.print(f"[yellow]Hint: {param_name} must be an integer[/yellow]")
            else:
                console.print(f"[yellow]Hint: {param_name} must be a number[/yellow]")
            raise typer.Exit(code=1) from None

        # 4. Load and validate data
        console.print("[cyan]Loading data...[/cyan]")
        data_path = Path(data)

        if not data_path.exists():
            console.print(f"[red]Error: File not found - {data}[/red]")
            raise typer.Exit(code=1)

        if data_path.suffix.lower() in [".parquet", ".pq"]:
            df = load_parquet(data_path)
        else:
            df = load_csv(data_path)
        console.print(f"[green]Loaded {len(df)} bars[/green]")

        # 5. Extract OHLCV arrays
        open_arr = df["open"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        close_arr = df["close"].values
        volume_arr = df["volume"].values

        # 6. Get generator function from registry and generate bars
        console.print(f"[cyan]Generating {bar_type} bars with {param_name}={param_value}...[/cyan]")
        generator = get_bar_generator(bar_type)
        kwargs = {param_name: param_value}
        bars = generator(open_arr, high_arr, low_arr, close_arr, volume_arr, **kwargs)

        # 7. Calculate and display stats
        source_count = len(df)
        bar_count = len(bars)
        compression_ratio = source_count / bar_count if bar_count > 0 else 0.0

        console.print(f"[green]Generated {bar_count} bars from {source_count} source bars[/green]")
        console.print(
            f"[cyan]Compression ratio: {compression_ratio:.2f}:1 "
            f"({source_count} -> {bar_count} bars)[/cyan]"
        )

        # 8. Determine output path
        if output:
            output_path = Path(output)
        else:
            output_path = data_path.parent / f"{data_path.stem}_{bar_type}_bars.csv"

        # 9. Create output DataFrame with source datetime mapping
        source_datetime = df["datetime"].values
        bar_datetime = source_datetime[bars.index_map]

        output_df = pd.DataFrame(
            {
                "datetime": bar_datetime,
                "open": bars.open,
                "high": bars.high,
                "low": bars.low,
                "close": bars.close,
                "volume": bars.volume,
                "source_index": bars.index_map,
            }
        )

        # 10. Ensure output directory exists and save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        console.print(f"[green]Saved {bar_count} bars to {output_path}[/green]")

    except typer.Exit:
        raise
    except KeyError as e:
        console.print(f"[red]Bar type error: {e}[/red]")
        available_types = list_bar_types()
        if available_types:
            console.print(f"Available types: {', '.join(available_types)}")
        raise typer.Exit(code=1) from None
    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(code=1) from None
    except DataLoadError as e:
        console.print(f"[red]Data validation error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except ValueError as e:
        console.print(f"[red]Parameter error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None


@app.command()
def benchmark(
    suite: str = typer.Option(
        "full", "--suite", "-s", help="Benchmark suite: full, bars, backtest, indicators"
    ),
) -> None:
    """Run performance benchmarks and display results vs targets.

    Executes pytest-benchmark for the specified suite and displays
    throughput/latency results compared against performance targets.

    Suites:
        - full: All benchmark categories
        - bars: Bar generation benchmarks only
        - backtest: Backtest engine benchmarks only
        - indicators: Indicator benchmarks only

    Example:
        sfb benchmark --suite full
        sfb benchmark --suite bars
    """
    from simple_futures_backtester.utils.benchmarks import (
        SUITE_MAP,
        check_all_passed,
        create_benchmark_table,
        get_available_suites,
        parse_benchmark_output,
        run_benchmark_suite,
    )

    # Validate suite option
    available_suites = get_available_suites()
    if suite not in available_suites:
        console.print(f"[red]Error: Unknown suite '{suite}'[/red]")
        console.print(f"Available suites: {', '.join(available_suites)}")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Running benchmarks: {suite}[/bold blue]")
    console.print(f'[dim]Marker expression: -m "{SUITE_MAP[suite]}"[/dim]\n')

    try:
        # Run benchmark suite
        exit_code, stdout, stderr = run_benchmark_suite(suite)

        # Check for no benchmarks found
        if "No benchmark tests found" in stdout or "No benchmark" in stdout:
            console.print(f"[yellow]{stdout.strip()}[/yellow]")
            console.print("\n[dim]Create benchmark tests in tests/benchmarks/ with:")
            console.print("  - Filename pattern: bench_*.py")
            console.print("  - Use @pytest.mark.benchmark decorator[/dim]")
            raise typer.Exit(code=0)

        # Check for errors
        if stderr and "error" in stderr.lower():
            console.print(f"[red]Benchmark error:[/red]\n{stderr}")
            raise typer.Exit(code=1)

        # Parse results
        results = parse_benchmark_output(stdout)

        if not results:
            # No structured results, show raw output
            console.print("[yellow]No benchmark results could be parsed.[/yellow]")
            console.print("\n[dim]Raw output:[/dim]")
            console.print(stdout[:2000] if len(stdout) > 2000 else stdout)
            raise typer.Exit(code=exit_code)

        # Create and display results table
        table = create_benchmark_table(results, title=f"Benchmark Results ({suite})")
        console.print(table)

        # Summary
        passed_count = sum(1 for r in results if check_all_passed([r]))
        total_count = len(results)

        console.print(f"\n[bold]Summary:[/bold] {passed_count}/{total_count} benchmarks passed")

        # Determine exit code based on pass/fail
        if check_all_passed(results):
            console.print("[green]All benchmarks passed![/green]")
            raise typer.Exit(code=0)
        else:
            console.print("[red]Some benchmarks failed to meet targets.[/red]")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Benchmark error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None


@app.command()
def export(
    input_file: str = typer.Option(..., "--input", "-i", help="Input results JSON file"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory path"),
    format: str = typer.Option("all", "--format", "-f", help="Export format: all, png, html, csv"),
) -> None:
    """Export backtest results to various formats.

    Re-exports results from a previous backtest JSON report to PNG charts,
    HTML interactive charts, or CSV data files.

    Formats:
        - all: Export all formats (PNG, HTML, CSV)
        - png: Static PNG chart images only
        - html: Interactive HTML charts only
        - csv: CSV data files only

    Example:
        sfb export --input results.json --output ./exports --format all
        sfb export --input results.json --output ./charts --format png
    """
    import json as json_module

    import numpy as np
    import pandas as pd

    from simple_futures_backtester.backtest.engine import BacktestResult
    from simple_futures_backtester.output.exports import ResultsExporter

    # Valid formats
    VALID_FORMATS = ["all", "png", "html", "csv"]

    try:
        # Validate format option
        if format not in VALID_FORMATS:
            console.print(f"[red]Error: Unknown format '{format}'[/red]")
            console.print(f"Available formats: {', '.join(VALID_FORMATS)}")
            raise typer.Exit(code=1)

        # Validate input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]Error: Input file not found - {input_file}[/red]")
            raise typer.Exit(code=1)

        if input_path.suffix.lower() != ".json":
            console.print(f"[yellow]Warning: Expected .json file, got {input_path.suffix}[/yellow]")

        console.print(f"[cyan]Loading results from {input_file}...[/cyan]")

        # Load JSON report
        with open(input_path) as f:
            data = json_module.load(f)

        # Validate JSON structure
        required_keys = ["metrics", "equity_curve", "drawdown_curve"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            console.print(f"[red]Error: Invalid results file. Missing keys: {missing_keys}[/red]")
            console.print("[dim]Expected JSON structure from backtest --output[/dim]")
            raise typer.Exit(code=1)

        # Reconstruct BacktestResult from JSON
        metrics = data["metrics"]

        # Handle trades - convert list of dicts to DataFrame
        trades_data = data.get("trades", [])
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "Entry Time",
                    "Exit Time",
                    "Entry Price",
                    "Exit Price",
                    "PnL",
                    "Return",
                    "Duration",
                    "Direction",
                ]
            )

        result = BacktestResult(
            total_return=float(metrics.get("total_return", 0)),
            sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
            sortino_ratio=float(metrics.get("sortino_ratio", 0)),
            max_drawdown=float(metrics.get("max_drawdown", 0)),
            win_rate=float(metrics.get("win_rate", 0)),
            profit_factor=float(metrics.get("profit_factor", 0)),
            n_trades=int(data.get("trades_count", metrics.get("n_trades", 0))),
            avg_trade=float(metrics.get("avg_trade", 0)),
            equity_curve=np.array(data["equity_curve"], dtype=np.float64),
            drawdown_curve=np.array(data["drawdown_curve"], dtype=np.float64),
            trades=trades_df,
            config_hash=data.get("config_hash", ""),
            timestamp=data.get("timestamp", ""),
        )

        console.print(f"[green]Loaded backtest results ({result.n_trades} trades)[/green]")

        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Exporting to {output_path}/ (format: {format})...[/cyan]")

        # Export based on format
        # Note: close_prices not available from JSON, so trades chart won't be created
        if format == "all":
            # Full export with organized structure
            ResultsExporter.export_all(
                result,
                output_dir=output_path,
                strategy_name=data.get("strategy_name", "exported"),
                close_prices=None,  # Not available from JSON
                dark_theme=False,
            )
            console.print("[green]Exported: charts/equity.png, drawdown.png, monthly.png[/green]")
            console.print(
                "[green]Exported: charts/equity.html, drawdown.html, monthly.html[/green]"
            )
            console.print(
                "[green]Exported: data/equity_curve.csv, trades.csv, metrics.csv, monthly_returns.csv[/green]"
            )
            console.print("[green]Exported: report.json[/green]")

        elif format == "png":
            ResultsExporter.export_charts_png(
                result,
                output_dir=output_path,
                close_prices=None,
            )
            console.print("[green]Exported: equity.png, drawdown.png, monthly.png[/green]")

        elif format == "html":
            ResultsExporter.export_charts_html(
                result,
                output_dir=output_path,
                close_prices=None,
            )
            console.print("[green]Exported: equity.html, drawdown.html, monthly.html[/green]")

        elif format == "csv":
            ResultsExporter.export_csv(result, output_dir=output_path)
            console.print(
                "[green]Exported: equity_curve.csv, trades.csv, metrics.csv, monthly_returns.csv[/green]"
            )

        console.print(f"\n[bold green]Export complete![/bold green] Files saved to {output_path}/")
        console.print("[dim]Note: trades chart not exported (requires original price data)[/dim]")

    except typer.Exit:
        raise
    except json_module.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in input file - {e}[/red]")
        raise typer.Exit(code=1) from None
    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(code=1) from None
    except KeyError as e:
        console.print(f"[red]Error: Missing required field in results file - {e}[/red]")
        raise typer.Exit(code=1) from None
    except PermissionError as e:
        console.print(f"[red]Error: Permission denied - {e}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Export error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from None


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
