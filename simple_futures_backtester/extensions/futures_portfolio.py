"""FuturesPortfolio wrapper for VectorBT Portfolio with point value support.

Provides a wrapper class that applies futures-specific point value multipliers
to convert raw price-based PnL metrics into dollar-denominated values. This
wrapper does NOT modify the backtest execution; it only transforms metrics
at extraction time.

Futures contracts have a point value that represents the dollar amount per
point of price movement. For example:
- ES (E-mini S&P 500): $50 per point
- NQ (E-mini NASDAQ-100): $20 per point
- GC (Gold): $100 per point

When VectorBT runs a backtest on raw prices (e.g., ES at 4500.00), the PnL
is expressed in points. This wrapper multiplies those values by point_value
to get actual dollar PnL.

Usage:
    >>> import vectorbt as vbt
    >>> from simple_futures_backtester.extensions.futures_portfolio import (
    ...     FuturesPortfolio,
    ...     PortfolioAnalytics,
    ... )
    >>>
    >>> # Run backtest with VectorBT on raw prices
    >>> portfolio = vbt.Portfolio.from_signals(
    ...     close=price_data,
    ...     entries=entry_signals,
    ...     exits=exit_signals,
    ...     init_cash=100000.0,
    ...     fees=0.0001,
    ...     freq='1D',
    ... )
    >>>
    >>> # Wrap with futures specifications (e.g., ES contract)
    >>> futures_pf = FuturesPortfolio(
    ...     portfolio=portfolio,
    ...     point_value=50.0,  # $50 per point for ES
    ...     tick_size=0.25,    # ES trades in 0.25 increments
    ... )
    >>>
    >>> # Get dollar-denominated analytics
    >>> analytics = futures_pf.get_analytics()
    >>> print(f"Total PnL: ${analytics.total_pnl:,.2f}")
    >>> print(f"Sharpe Ratio: {analytics.sharpe_ratio:.2f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import vectorbt as vbt


@dataclass
class PortfolioAnalytics:
    """Portfolio analytics with futures-specific dollar-denominated metrics.

    All PnL metrics are expressed in dollars by applying the futures point_value
    multiplier. Ratios and percentages are left in their natural units.

    Attributes:
        total_pnl: Total profit/loss in dollars.
        avg_trade_pnl: Average trade profit/loss in dollars.
        avg_win_dollars: Average winning trade in dollars.
        avg_loss_dollars: Average losing trade in dollars (positive number).
        max_win_dollars: Largest winning trade in dollars.
        max_loss_dollars: Largest losing trade in dollars (positive number).
        max_drawdown_dollars: Maximum drawdown in dollars.
        sharpe_ratio: Sharpe ratio (risk-adjusted return, dimensionless).
        sortino_ratio: Sortino ratio (downside-adjusted return, dimensionless).
        calmar_ratio: Calmar ratio (return/max drawdown, dimensionless).
        profit_factor: Ratio of gross profits to gross losses.
        total_return: Total return as decimal (0.15 = 15% return).
        max_drawdown_percent: Maximum drawdown as decimal (0.20 = 20% drawdown).
        win_rate: Winning trade percentage as decimal (0.60 = 60% win rate).
        n_trades: Total number of trades.
        n_wins: Number of winning trades.
        n_losses: Number of losing trades.
        point_value: Futures point value used for dollar conversion.
        tick_size: Minimum price increment (for price formatting).
        total_fees: Total fees paid in dollars.
        expectancy: Expected value per trade in dollars.

    Example:
        >>> analytics = futures_pf.get_analytics()
        >>> print(f"Total PnL: ${analytics.total_pnl:,.2f}")
        >>> print(f"Sharpe Ratio: {analytics.sharpe_ratio:.2f}")
        >>> print(f"Win Rate: {analytics.win_rate:.1%}")
    """

    # Dollar-denominated metrics
    total_pnl: float
    avg_trade_pnl: float
    avg_win_dollars: float
    avg_loss_dollars: float
    max_win_dollars: float
    max_loss_dollars: float
    max_drawdown_dollars: float

    # Ratio metrics (dimensionless, no point_value multiplication)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Percentage metrics (as decimals, e.g., 0.15 = 15%)
    total_return: float
    max_drawdown_percent: float
    win_rate: float

    # Count metrics
    n_trades: int
    n_wins: int
    n_losses: int

    # Metadata
    point_value: float
    tick_size: float

    # Additional dollar metrics
    total_fees: float
    expectancy: float


class FuturesPortfolio:
    """Wrapper for VectorBT Portfolio that applies futures point value.

    This class wraps an existing VectorBT Portfolio object and provides
    dollar-denominated analytics by applying the futures contract's point
    value multiplier at metric extraction time.

    IMPORTANT: The wrapped portfolio should be run on raw prices (not
    pre-multiplied by point_value). The point value is applied ONLY when
    extracting metrics, not during the backtest simulation.

    Attributes:
        portfolio: The underlying VectorBT Portfolio object.
        point_value: Dollar value per point of price movement.
        tick_size: Minimum price increment for the contract.

    Example:
        >>> import vectorbt as vbt
        >>> from simple_futures_backtester.extensions import FuturesPortfolio
        >>>
        >>> # Create VectorBT portfolio from signals
        >>> pf = vbt.Portfolio.from_signals(
        ...     close=es_prices,
        ...     entries=signals,
        ...     exits=exits,
        ...     init_cash=100000.0,
        ... )
        >>>
        >>> # Wrap with ES futures specs
        >>> futures_pf = FuturesPortfolio(
        ...     portfolio=pf,
        ...     point_value=50.0,
        ...     tick_size=0.25,
        ... )
        >>>
        >>> # Get dollar-denominated metrics
        >>> analytics = futures_pf.get_analytics()
    """

    def __init__(
        self,
        portfolio: vbt.Portfolio,
        point_value: float,
        tick_size: float,
    ) -> None:
        """Initialize FuturesPortfolio wrapper.

        Args:
            portfolio: VectorBT Portfolio object (already executed).
                Must be created using Portfolio.from_signals(),
                Portfolio.from_orders(), or Portfolio.from_order_func().
            point_value: Dollar value per point of price movement.
                For example, 50.0 for ES (E-mini S&P 500).
            tick_size: Minimum price increment for the contract.
                For example, 0.25 for ES. Used for price formatting.

        Raises:
            ValueError: If point_value or tick_size is not positive.

        Example:
            >>> futures_pf = FuturesPortfolio(
            ...     portfolio=vbt_portfolio,
            ...     point_value=50.0,
            ...     tick_size=0.25,
            ... )
        """
        if point_value <= 0:
            raise ValueError(f"point_value must be positive, got {point_value}")
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")

        self._portfolio = portfolio
        self._point_value = point_value
        self._tick_size = tick_size

    @property
    def portfolio(self) -> vbt.Portfolio:
        """Access the underlying VectorBT Portfolio object."""
        return self._portfolio

    @property
    def point_value(self) -> float:
        """Get the futures point value (dollars per point)."""
        return self._point_value

    @property
    def tick_size(self) -> float:
        """Get the minimum price increment (tick size)."""
        return self._tick_size

    def format_price(self, price: float) -> float:
        """Round price to the nearest tick size.

        Args:
            price: Raw price value.

        Returns:
            Price rounded to nearest tick increment.

        Example:
            >>> futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)
            >>> futures_pf.format_price(4500.13)  # Returns 4500.25
        """
        return round(price / self._tick_size) * self._tick_size

    def _safe_float(self, value: float, default: float = 0.0) -> float:
        """Safely convert value to float, handling NaN and None.

        Args:
            value: Value to convert.
            default: Default value if conversion fails or value is NaN.

        Returns:
            Float value or default if invalid.
        """
        if value is None:
            return default
        try:
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except (TypeError, ValueError):
            return default

    def get_analytics(self) -> PortfolioAnalytics:
        """Extract portfolio analytics with dollar-denominated PnL metrics.

        This method extracts metrics from the underlying VectorBT Portfolio
        and applies the point value multiplier to all PnL-related fields.
        Ratio metrics (Sharpe, Sortino, etc.) are NOT multiplied as they
        are dimensionless.

        Returns:
            PortfolioAnalytics dataclass with all metrics populated.
            Dollar fields are multiplied by point_value.
            Percentages are converted to decimals (0.15 = 15%).

        Example:
            >>> analytics = futures_pf.get_analytics()
            >>> print(f"Total PnL: ${analytics.total_pnl:,.2f}")
            >>> print(f"Sharpe Ratio: {analytics.sharpe_ratio:.2f}")
            >>> print(f"Max Drawdown: {analytics.max_drawdown_percent:.1%}")
        """
        pf = self._portfolio
        pv = self._point_value

        # Extract total profit (in points) and convert to dollars
        # VectorBT total_profit() returns profit in price units
        total_profit_points = self._safe_float(pf.total_profit())
        total_pnl = total_profit_points * pv

        # Extract total return as decimal
        # VectorBT total_return() returns as decimal (0.15 = 15%)
        total_return = self._safe_float(pf.total_return())

        # Get trade records for win/loss calculations
        trades = pf.trades
        closed_trades = trades.closed if hasattr(trades, "closed") else trades

        # Trade counts
        n_trades = int(self._safe_float(closed_trades.count()))

        # Winning/losing trade counts
        if hasattr(closed_trades, "winning") and hasattr(closed_trades, "losing"):
            n_wins = int(self._safe_float(closed_trades.winning.count()))
            n_losses = int(self._safe_float(closed_trades.losing.count()))
        else:
            # Fallback: calculate from win rate
            win_rate_pct = (
                self._safe_float(closed_trades.win_rate())
                if hasattr(closed_trades, "win_rate")
                else 0.0
            )
            n_wins = int(round(n_trades * win_rate_pct)) if n_trades > 0 else 0
            n_losses = n_trades - n_wins

        # Win rate as decimal
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        # Average trade PnL in dollars
        if n_trades > 0 and hasattr(closed_trades, "pnl"):
            avg_trade_pnl_points = self._safe_float(closed_trades.pnl.mean())
            avg_trade_pnl = avg_trade_pnl_points * pv
        else:
            avg_trade_pnl = total_pnl / n_trades if n_trades > 0 else 0.0

        # Average winning trade in dollars
        if hasattr(closed_trades, "winning") and hasattr(closed_trades.winning, "pnl"):
            avg_win_points = self._safe_float(closed_trades.winning.pnl.mean())
            avg_win_dollars = avg_win_points * pv
        else:
            avg_win_dollars = 0.0

        # Average losing trade in dollars (stored as positive number)
        if hasattr(closed_trades, "losing") and hasattr(closed_trades.losing, "pnl"):
            avg_loss_points = self._safe_float(closed_trades.losing.pnl.mean())
            avg_loss_dollars = abs(avg_loss_points * pv)
        else:
            avg_loss_dollars = 0.0

        # Max winning trade in dollars
        if hasattr(closed_trades, "pnl"):
            max_win_points = self._safe_float(closed_trades.pnl.max())
            max_win_dollars = max_win_points * pv if max_win_points > 0 else 0.0
        else:
            max_win_dollars = 0.0

        # Max losing trade in dollars (stored as positive number)
        if hasattr(closed_trades, "pnl"):
            min_pnl_points = self._safe_float(closed_trades.pnl.min())
            max_loss_dollars = abs(min_pnl_points * pv) if min_pnl_points < 0 else 0.0
        else:
            max_loss_dollars = 0.0

        # Max drawdown percentage (convert from VectorBT format)
        # VectorBT drawdowns.max_drawdown returns as decimal (negative)
        if hasattr(pf, "drawdowns") and hasattr(pf.drawdowns, "max_drawdown"):
            max_dd_raw = self._safe_float(pf.drawdowns.max_drawdown())
            max_drawdown_percent = abs(max_dd_raw)  # Store as positive decimal
        else:
            max_drawdown_percent = 0.0

        # Max drawdown in dollars
        # Calculated from initial capital and max drawdown percentage
        init_cash = self._safe_float(pf.init_cash) if hasattr(pf, "init_cash") else 100000.0
        max_drawdown_dollars = init_cash * max_drawdown_percent

        # Profit factor (dimensionless ratio)
        if hasattr(closed_trades, "profit_factor"):
            profit_factor = self._safe_float(closed_trades.profit_factor(), default=0.0)
        else:
            profit_factor = 0.0

        # Expectancy in dollars
        if hasattr(closed_trades, "expectancy"):
            expectancy_points = self._safe_float(closed_trades.expectancy())
            expectancy = expectancy_points * pv
        else:
            expectancy = avg_trade_pnl

        # Total fees in dollars
        if hasattr(pf, "orders") and hasattr(pf.orders, "fees"):
            total_fees_points = self._safe_float(pf.orders.fees.sum())
            total_fees = total_fees_points * pv
        else:
            total_fees = 0.0

        # Risk-adjusted ratios (dimensionless, no point_value multiplication)
        returns_acc = pf.returns_acc if hasattr(pf, "returns_acc") else None

        if returns_acc is not None:
            sharpe_ratio = self._safe_float(returns_acc.sharpe_ratio())
            sortino_ratio = self._safe_float(returns_acc.sortino_ratio())
            calmar_ratio = self._safe_float(returns_acc.calmar_ratio())
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0

        return PortfolioAnalytics(
            # Dollar metrics
            total_pnl=total_pnl,
            avg_trade_pnl=avg_trade_pnl,
            avg_win_dollars=avg_win_dollars,
            avg_loss_dollars=avg_loss_dollars,
            max_win_dollars=max_win_dollars,
            max_loss_dollars=max_loss_dollars,
            max_drawdown_dollars=max_drawdown_dollars,
            # Ratio metrics (dimensionless)
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            # Percentage metrics (as decimals)
            total_return=total_return,
            max_drawdown_percent=max_drawdown_percent,
            win_rate=win_rate,
            # Count metrics
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            # Metadata
            point_value=self._point_value,
            tick_size=self._tick_size,
            # Additional dollar metrics
            total_fees=total_fees,
            expectancy=expectancy,
        )

    def get_equity_curve(self) -> np.ndarray:
        """Get the equity curve in dollars.

        Returns:
            NumPy array of portfolio value at each bar, in dollars.
            The value is multiplied by point_value to convert from
            price units to dollars.

        Example:
            >>> equity = futures_pf.get_equity_curve()
            >>> plt.plot(equity)
        """
        value = self._portfolio.value()
        if hasattr(value, "values"):
            value = value.values
        return np.asarray(value) * self._point_value

    def get_drawdown_curve(self) -> np.ndarray:
        """Get the drawdown curve as percentages (decimals).

        Returns:
            NumPy array of drawdown at each bar as decimal (0.15 = 15%).
            Values are negative or zero (0.0 = no drawdown, -0.15 = 15% dd).

        Example:
            >>> drawdown = futures_pf.get_drawdown_curve()
            >>> max_dd = drawdown.min()  # Most negative = worst drawdown
        """
        # Calculate from equity curve directly
        # VectorBT's drawdowns.drawdown returns individual drawdown records,
        # not a time series, so we compute it manually
        equity = self._portfolio.value()
        if hasattr(equity, "values"):
            equity = equity.values
        equity = np.asarray(equity)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown


__all__: list[str] = [
    "FuturesPortfolio",
    "PortfolioAnalytics",
]
