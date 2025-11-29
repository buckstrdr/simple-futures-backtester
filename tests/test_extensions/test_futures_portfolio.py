"""Tests for FuturesPortfolio wrapper class.

Tests cover:
- Initialization and validation (point_value, tick_size)
- Point value application to dollar-denominated metrics
- Analytics method verification and dataclass field population
- Edge cases (zero trades, NaN/None/inf handling)
- Equity and drawdown curve extraction
- Tick size formatting
- _safe_float helper method

Performance Notes:
    FuturesPortfolio is a wrapper class that adds <5% overhead vs direct
    VectorBT Portfolio access. The wrapper does NOT re-run backtests; it
    only transforms metrics at extraction time by applying the point_value
    multiplier to dollar-denominated fields.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from simple_futures_backtester.extensions.futures_portfolio import (
    FuturesPortfolio,
    PortfolioAnalytics,
)


class TestFuturesPortfolioInitialization:
    """Tests for FuturesPortfolio constructor and validation."""

    def test_valid_initialization(self) -> None:
        """Should initialize successfully with valid parameters."""
        # Arrange
        portfolio = mock.Mock()

        # Act
        futures_pf = FuturesPortfolio(
            portfolio=portfolio,
            point_value=50.0,
            tick_size=0.25,
        )

        # Assert
        assert futures_pf.portfolio is portfolio
        assert futures_pf.point_value == 50.0
        assert futures_pf.tick_size == 0.25

    def test_point_value_validation_raises_error_on_zero(self) -> None:
        """Should raise ValueError when point_value is zero."""
        # Arrange
        portfolio = mock.Mock()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FuturesPortfolio(
                portfolio=portfolio,
                point_value=0.0,
                tick_size=0.25,
            )

        assert "point_value must be positive" in str(exc_info.value)

    def test_point_value_validation_raises_error_on_negative(self) -> None:
        """Should raise ValueError when point_value is negative."""
        # Arrange
        portfolio = mock.Mock()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FuturesPortfolio(
                portfolio=portfolio,
                point_value=-50.0,
                tick_size=0.25,
            )

        assert "point_value must be positive" in str(exc_info.value)

    def test_tick_size_validation_raises_error_on_zero(self) -> None:
        """Should raise ValueError when tick_size is zero."""
        # Arrange
        portfolio = mock.Mock()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FuturesPortfolio(
                portfolio=portfolio,
                point_value=50.0,
                tick_size=0.0,
            )

        assert "tick_size must be positive" in str(exc_info.value)

    def test_tick_size_validation_raises_error_on_negative(self) -> None:
        """Should raise ValueError when tick_size is negative."""
        # Arrange
        portfolio = mock.Mock()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FuturesPortfolio(
                portfolio=portfolio,
                point_value=50.0,
                tick_size=-0.25,
            )

        assert "tick_size must be positive" in str(exc_info.value)

    def test_properties_accessible(self) -> None:
        """Should access portfolio, point_value, and tick_size via properties."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(
            portfolio=portfolio,
            point_value=50.0,
            tick_size=0.25,
        )

        # Act & Assert
        assert futures_pf.portfolio is portfolio
        assert futures_pf.point_value == 50.0
        assert futures_pf.tick_size == 0.25


class TestPointValueApplication:
    """Tests verifying point value multiplier applied correctly."""

    @pytest.mark.parametrize("point_value", [1.0, 2.0, 50.0])
    def test_total_pnl_multiplied_by_point_value(self, point_value: float) -> None:
        """Total PnL should be multiplied by point_value."""
        # Arrange: Mock portfolio with total_profit in price units
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0  # 100 points profit
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        # Mock trades (minimal)
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock returns_acc (minimal)
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        expected_pnl = 100.0 * point_value
        assert abs(analytics.total_pnl - expected_pnl) < 1e-10

    @pytest.mark.parametrize("point_value", [1.0, 2.0, 50.0])
    def test_avg_trade_pnl_multiplied(self, point_value: float) -> None:
        """Average trade PnL should be multiplied by point_value."""
        # Arrange
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        # Mock trades with avg PnL
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10
        closed.pnl = mock.Mock()
        closed.pnl.mean.return_value = 10.0  # 10 points per trade
        closed.winning = mock.Mock()
        closed.winning.count.return_value = 6
        closed.losing = mock.Mock()
        closed.losing.count.return_value = 4
        trades.closed = closed
        portfolio.trades = trades

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        expected_avg = 10.0 * point_value
        assert abs(analytics.avg_trade_pnl - expected_avg) < 1e-10

    @pytest.mark.parametrize("point_value", [1.0, 2.0, 50.0])
    def test_max_drawdown_dollars_multiplied(self, point_value: float) -> None:
        """Max drawdown in dollars should be calculated from init_cash and percentage."""
        # Arrange
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 0.0
        portfolio.total_return.return_value = 0.0
        portfolio.init_cash = 100000.0

        # Mock trades
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.20  # 20% drawdown
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 0.0
        returns_acc.sortino_ratio.return_value = 0.0
        returns_acc.calmar_ratio.return_value = 0.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: max_drawdown_dollars = init_cash * abs(max_dd_percent)
        expected_dd_dollars = 100000.0 * 0.20
        assert abs(analytics.max_drawdown_dollars - expected_dd_dollars) < 1e-10
        assert abs(analytics.max_drawdown_percent - 0.20) < 1e-10

    @pytest.mark.parametrize("point_value", [1.0, 2.0, 50.0])
    def test_ratio_metrics_not_multiplied(self, point_value: float) -> None:
        """Ratio metrics (Sharpe, Sortino, Calmar, profit_factor) should NOT be multiplied."""
        # Arrange
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        # Mock trades with profit_factor
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10
        closed.profit_factor.return_value = 1.5  # dimensionless ratio
        closed.winning = mock.Mock()
        closed.winning.count.return_value = 6
        closed.losing = mock.Mock()
        closed.losing.count.return_value = 4
        trades.closed = closed
        portfolio.trades = trades

        # Mock returns_acc with ratios
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.2
        returns_acc.sortino_ratio.return_value = 1.5
        returns_acc.calmar_ratio.return_value = 0.8
        portfolio.returns_acc = returns_acc

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Ratios should be unchanged regardless of point_value
        assert abs(analytics.sharpe_ratio - 1.2) < 1e-10
        assert abs(analytics.sortino_ratio - 1.5) < 1e-10
        assert abs(analytics.calmar_ratio - 0.8) < 1e-10
        assert abs(analytics.profit_factor - 1.5) < 1e-10

    def test_percentage_metrics_preserved(self) -> None:
        """Percentage metrics should be stored as decimals (0.15 = 15%)."""
        # Arrange
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 15000.0
        portfolio.total_return.return_value = 0.15  # 15% return
        portfolio.init_cash = 100000.0

        # Mock trades
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10
        closed.winning = mock.Mock()
        closed.winning.count.return_value = 6
        closed.losing = mock.Mock()
        closed.losing.count.return_value = 4
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.10  # 10% drawdown
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        assert abs(analytics.total_return - 0.15) < 1e-10
        assert abs(analytics.max_drawdown_percent - 0.10) < 1e-10
        assert abs(analytics.win_rate - 0.60) < 1e-10  # 6/10


class TestGetAnalytics:
    """Tests for get_analytics() method."""

    def test_analytics_dataclass_returned(self) -> None:
        """get_analytics() should return PortfolioAnalytics dataclass instance."""
        # Arrange: Minimal mock portfolio
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        assert isinstance(analytics, PortfolioAnalytics)

    def test_all_fields_populated(self) -> None:
        """All 19 fields in PortfolioAnalytics should be populated."""
        # Arrange: Complete mock portfolio
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 5000.0
        portfolio.total_return.return_value = 0.05
        portfolio.init_cash = 100000.0

        # Mock trades with all attributes
        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 100

        pnl_mock = mock.Mock()
        pnl_mock.mean.return_value = 50.0
        pnl_mock.max.return_value = 200.0
        pnl_mock.min.return_value = -100.0
        closed.pnl = pnl_mock

        winning = mock.Mock()
        winning.count.return_value = 60
        winning_pnl = mock.Mock()
        winning_pnl.mean.return_value = 100.0
        winning.pnl = winning_pnl
        closed.winning = winning

        losing = mock.Mock()
        losing.count.return_value = 40
        losing_pnl = mock.Mock()
        losing_pnl.mean.return_value = -50.0
        losing.pnl = losing_pnl
        closed.losing = losing

        closed.profit_factor.return_value = 2.0
        closed.expectancy.return_value = 50.0

        trades.closed = closed
        portfolio.trades = trades

        # Mock orders
        orders = mock.Mock()
        fees_mock = mock.Mock()
        fees_mock.sum.return_value = 100.0
        orders.fees = fees_mock
        portfolio.orders = orders

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.10
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.5
        returns_acc.sortino_ratio.return_value = 1.8
        returns_acc.calmar_ratio.return_value = 0.5
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: All fields should be populated (19 fields)
        assert hasattr(analytics, "total_pnl")
        assert hasattr(analytics, "avg_trade_pnl")
        assert hasattr(analytics, "avg_win_dollars")
        assert hasattr(analytics, "avg_loss_dollars")
        assert hasattr(analytics, "max_win_dollars")
        assert hasattr(analytics, "max_loss_dollars")
        assert hasattr(analytics, "max_drawdown_dollars")
        assert hasattr(analytics, "sharpe_ratio")
        assert hasattr(analytics, "sortino_ratio")
        assert hasattr(analytics, "calmar_ratio")
        assert hasattr(analytics, "profit_factor")
        assert hasattr(analytics, "total_return")
        assert hasattr(analytics, "max_drawdown_percent")
        assert hasattr(analytics, "win_rate")
        assert hasattr(analytics, "n_trades")
        assert hasattr(analytics, "n_wins")
        assert hasattr(analytics, "n_losses")
        assert hasattr(analytics, "point_value")
        assert hasattr(analytics, "tick_size")
        assert hasattr(analytics, "total_fees")
        assert hasattr(analytics, "expectancy")

        # Verify specific values
        assert analytics.n_trades == 100
        assert analytics.n_wins == 60
        assert analytics.n_losses == 40
        assert abs(analytics.win_rate - 0.60) < 1e-10
        assert analytics.point_value == 50.0
        assert analytics.tick_size == 0.25

    def test_dollar_metrics_vs_price_metrics(self) -> None:
        """Dollar metrics should equal price metrics * point_value."""
        # Arrange
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0  # 100 points
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10

        pnl_mock = mock.Mock()
        pnl_mock.mean.return_value = 10.0  # 10 points per trade
        pnl_mock.max.return_value = 50.0  # 50 points max win
        pnl_mock.min.return_value = -20.0  # -20 points max loss
        closed.pnl = pnl_mock

        winning = mock.Mock()
        winning.count.return_value = 6
        winning_pnl = mock.Mock()
        winning_pnl.mean.return_value = 25.0  # 25 points avg win
        winning.pnl = winning_pnl
        closed.winning = winning

        losing = mock.Mock()
        losing.count.return_value = 4
        losing_pnl = mock.Mock()
        losing_pnl.mean.return_value = -10.0  # -10 points avg loss
        losing.pnl = losing_pnl
        closed.losing = losing

        closed.profit_factor.return_value = 1.5
        closed.expectancy.return_value = 10.0  # 10 points expectancy

        trades.closed = closed
        portfolio.trades = trades

        # Mock orders
        orders = mock.Mock()
        fees_mock = mock.Mock()
        fees_mock.sum.return_value = 5.0  # 5 points fees
        orders.fees = fees_mock
        portfolio.orders = orders

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.2
        returns_acc.sortino_ratio.return_value = 1.5
        returns_acc.calmar_ratio.return_value = 0.8
        portfolio.returns_acc = returns_acc

        point_value = 50.0
        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Dollar values should equal price values * point_value
        assert abs(analytics.total_pnl - 100.0 * point_value) < 1e-10
        assert abs(analytics.avg_trade_pnl - 10.0 * point_value) < 1e-10
        assert abs(analytics.avg_win_dollars - 25.0 * point_value) < 1e-10
        assert abs(analytics.avg_loss_dollars - abs(-10.0) * point_value) < 1e-10
        assert abs(analytics.max_win_dollars - 50.0 * point_value) < 1e-10
        assert abs(analytics.max_loss_dollars - abs(-20.0) * point_value) < 1e-10
        assert abs(analytics.total_fees - 5.0 * point_value) < 1e-10
        assert abs(analytics.expectancy - 10.0 * point_value) < 1e-10

    def test_fallback_logic_for_win_count(self) -> None:
        """Should calculate win count from win_rate when .winning.count() unavailable."""
        # Arrange: Portfolio without .winning attribute
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10
        # No .winning or .losing attributes
        delattr(closed, "winning")
        delattr(closed, "losing")

        # Provide win_rate method
        closed.win_rate.return_value = 0.60  # 60% win rate

        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Should calculate n_wins from win_rate
        assert analytics.n_trades == 10
        assert analytics.n_wins == 6  # 10 * 0.60 = 6
        assert analytics.n_losses == 4  # 10 - 6 = 4


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_trades_returns_defaults(self) -> None:
        """Should return sensible defaults when portfolio has zero trades."""
        # Arrange: Portfolio with no trades
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 0.0
        portfolio.total_return.return_value = 0.0
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0  # Zero trades
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = 0.0
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 0.0
        returns_acc.sortino_ratio.return_value = 0.0
        returns_acc.calmar_ratio.return_value = 0.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Should have zero values, not crash
        assert analytics.n_trades == 0
        assert analytics.n_wins == 0
        assert analytics.n_losses == 0
        assert analytics.total_pnl == 0.0
        assert analytics.avg_trade_pnl == 0.0
        assert analytics.win_rate == 0.0

    def test_nan_values_handled(self) -> None:
        """Should handle NaN values from VectorBT gracefully."""
        # Arrange: Portfolio returning NaN
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = float("nan")
        portfolio.total_return.return_value = float("nan")
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns with NaN
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = float("nan")
        portfolio.drawdowns = drawdowns

        # Mock returns_acc with NaN
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = float("nan")
        returns_acc.sortino_ratio.return_value = float("nan")
        returns_acc.calmar_ratio.return_value = float("nan")
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: NaN should convert to 0.0 (default)
        assert analytics.total_pnl == 0.0
        assert analytics.total_return == 0.0
        assert analytics.sharpe_ratio == 0.0
        assert analytics.max_drawdown_percent == 0.0

    def test_none_values_handled(self) -> None:
        """Should handle None values from VectorBT gracefully."""
        # Arrange: Portfolio returning None
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = None
        portfolio.total_return.return_value = None
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns with None
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = None
        portfolio.drawdowns = drawdowns

        # Mock returns_acc with None
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = None
        returns_acc.sortino_ratio.return_value = None
        returns_acc.calmar_ratio.return_value = None
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: None should convert to 0.0 (default)
        assert analytics.total_pnl == 0.0
        assert analytics.total_return == 0.0
        assert analytics.sharpe_ratio == 0.0
        assert analytics.max_drawdown_percent == 0.0

    def test_infinity_values_handled(self) -> None:
        """Should handle inf values from VectorBT gracefully."""
        # Arrange: Portfolio returning inf
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = float("inf")
        portfolio.total_return.return_value = float("-inf")
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = 0.0
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = float("inf")
        returns_acc.sortino_ratio.return_value = 0.0
        returns_acc.calmar_ratio.return_value = 0.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: inf should convert to 0.0 (default)
        assert analytics.total_pnl == 0.0
        assert analytics.total_return == 0.0
        assert analytics.sharpe_ratio == 0.0

    def test_all_winning_trades(self) -> None:
        """Should handle portfolio with all winning trades."""
        # Arrange: All trades are winners
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 1000.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10

        pnl_mock = mock.Mock()
        pnl_mock.mean.return_value = 100.0
        pnl_mock.max.return_value = 200.0
        pnl_mock.min.return_value = 50.0  # Minimum is positive
        closed.pnl = pnl_mock

        winning = mock.Mock()
        winning.count.return_value = 10
        winning_pnl = mock.Mock()
        winning_pnl.mean.return_value = 100.0
        winning.pnl = winning_pnl
        closed.winning = winning

        losing = mock.Mock()
        losing.count.return_value = 0
        closed.losing = losing

        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.05
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 2.0
        returns_acc.sortino_ratio.return_value = 2.5
        returns_acc.calmar_ratio.return_value = 2.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        assert analytics.n_trades == 10
        assert analytics.n_wins == 10
        assert analytics.n_losses == 0
        assert analytics.win_rate == 1.0
        assert analytics.max_loss_dollars == 0.0  # min_pnl is positive, so max_loss = 0

    def test_all_losing_trades(self) -> None:
        """Should handle portfolio with all losing trades."""
        # Arrange: All trades are losers
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = -1000.0
        portfolio.total_return.return_value = -0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 10

        pnl_mock = mock.Mock()
        pnl_mock.mean.return_value = -100.0
        pnl_mock.max.return_value = -50.0  # Maximum is negative
        pnl_mock.min.return_value = -200.0
        closed.pnl = pnl_mock

        winning = mock.Mock()
        winning.count.return_value = 0
        closed.winning = winning

        losing = mock.Mock()
        losing.count.return_value = 10
        losing_pnl = mock.Mock()
        losing_pnl.mean.return_value = -100.0
        losing.pnl = losing_pnl
        closed.losing = losing

        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.20
        portfolio.drawdowns = drawdowns

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = -1.0
        returns_acc.sortino_ratio.return_value = -1.5
        returns_acc.calmar_ratio.return_value = -0.5
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert
        assert analytics.n_trades == 10
        assert analytics.n_wins == 0
        assert analytics.n_losses == 10
        assert analytics.win_rate == 0.0
        assert analytics.max_win_dollars == 0.0  # max_pnl is negative, so max_win = 0

    def test_no_returns_acc_attribute(self) -> None:
        """Should handle missing returns_acc attribute gracefully."""
        # Arrange: Portfolio without returns_acc
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # Mock drawdowns
        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.10
        portfolio.drawdowns = drawdowns

        # No returns_acc attribute
        delattr(portfolio, "returns_acc")

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Ratio metrics should default to 0.0
        assert analytics.sharpe_ratio == 0.0
        assert analytics.sortino_ratio == 0.0
        assert analytics.calmar_ratio == 0.0

    def test_no_drawdowns_attribute(self) -> None:
        """Should handle missing drawdowns attribute gracefully."""
        # Arrange: Portfolio without drawdowns
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        # No drawdowns attribute
        delattr(portfolio, "drawdowns")

        # Mock returns_acc
        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Should not crash, max_drawdown_percent should be 0.0
        assert analytics.max_drawdown_percent == 0.0
        assert analytics.max_drawdown_dollars == 0.0

    def test_minimal_portfolio_missing_attributes(self) -> None:
        """Should handle portfolio with minimal attributes (missing pnl, profit_factor, etc.)."""
        # Arrange: Portfolio with minimal attributes
        portfolio = mock.Mock(spec=["total_profit", "total_return", "init_cash", "trades"])
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        # Trades without pnl, profit_factor, expectancy
        trades = mock.Mock()
        closed = mock.Mock(spec=["count"])
        closed.count.return_value = 10
        trades.closed = closed
        portfolio.trades = trades

        # Create futures portfolio
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        analytics = futures_pf.get_analytics()

        # Assert: Should use defaults for missing attributes
        assert analytics.n_trades == 10
        assert analytics.max_win_dollars == 0.0  # No pnl attribute
        assert analytics.max_loss_dollars == 0.0  # No pnl attribute
        assert analytics.profit_factor == 0.0  # No profit_factor attribute
        assert analytics.expectancy == 500.0  # Falls back to avg_trade_pnl (total_pnl / n_trades = 5000/10)
        assert analytics.total_fees == 0.0  # No orders attribute
        assert analytics.sharpe_ratio == 0.0  # No returns_acc attribute
        assert analytics.max_drawdown_percent == 0.0  # No drawdowns attribute


class TestEquityAndDrawdownCurves:
    """Tests for equity and drawdown curve extraction."""

    def test_get_equity_curve_multiplies_by_point_value(self) -> None:
        """Equity curve should be multiplied by point_value."""
        # Arrange: Mock portfolio with value() returning Series
        portfolio = mock.Mock()
        equity_values = pd.Series([100000.0, 100500.0, 101000.0, 100800.0])
        portfolio.value.return_value = equity_values

        point_value = 50.0
        futures_pf = FuturesPortfolio(portfolio, point_value, 0.25)

        # Act
        equity = futures_pf.get_equity_curve()

        # Assert: Should be multiplied by point_value
        expected = np.array([100000.0, 100500.0, 101000.0, 100800.0]) * point_value
        assert isinstance(equity, np.ndarray)
        assert len(equity) == 4
        assert np.allclose(equity, expected)

    def test_get_equity_curve_with_numpy_array(self) -> None:
        """Equity curve should handle NumPy array from value()."""
        # Arrange: Mock portfolio with value() returning ndarray
        portfolio = mock.Mock()
        equity_values = np.array([100000.0, 100500.0, 101000.0])
        portfolio.value.return_value = equity_values

        futures_pf = FuturesPortfolio(portfolio, 2.0, 0.25)

        # Act
        equity = futures_pf.get_equity_curve()

        # Assert
        expected = equity_values * 2.0
        assert isinstance(equity, np.ndarray)
        assert np.allclose(equity, expected)

    def test_get_drawdown_curve_returns_decimals(self) -> None:
        """Drawdown curve should return drawdowns as decimals (negative or zero)."""
        # Arrange: Mock portfolio with value() for drawdown calculation
        portfolio = mock.Mock()
        equity_values = pd.Series([100000.0, 105000.0, 103000.0, 106000.0, 102000.0])
        portfolio.value.return_value = equity_values

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        drawdown = futures_pf.get_drawdown_curve()

        # Assert: Drawdown should be negative or zero
        assert isinstance(drawdown, np.ndarray)
        assert len(drawdown) == 5
        assert drawdown[0] == 0.0  # No drawdown at start
        assert drawdown[1] == 0.0  # New high, no drawdown
        assert drawdown[2] < 0.0  # Drawdown from 105000 to 103000
        assert drawdown[3] == 0.0  # New high, no drawdown
        assert drawdown[4] < 0.0  # Drawdown from 106000 to 102000

        # Verify calculation: drawdown = (equity - running_max) / running_max
        running_max = np.maximum.accumulate(equity_values.values)
        expected_dd = (equity_values.values - running_max) / running_max
        assert np.allclose(drawdown, expected_dd)

    def test_get_drawdown_curve_with_ndarray_input(self) -> None:
        """Drawdown curve should handle equity as ndarray without .values attribute."""
        # Arrange: Mock portfolio with value() returning ndarray directly (no .values)
        portfolio = mock.Mock()
        equity_values = np.array([100000.0, 105000.0, 103000.0, 106000.0, 102000.0])
        portfolio.value.return_value = equity_values  # Return ndarray, not Series

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        drawdown = futures_pf.get_drawdown_curve()

        # Assert: Should handle ndarray input without error
        assert isinstance(drawdown, np.ndarray)
        assert len(drawdown) == 5

        # Verify calculation matches expected drawdown
        running_max = np.maximum.accumulate(equity_values)
        expected_dd = (equity_values - running_max) / running_max
        assert np.allclose(drawdown, expected_dd)

    def test_equity_curve_shape_matches_portfolio(self) -> None:
        """Equity curve shape should match VectorBT portfolio value."""
        # Arrange
        portfolio = mock.Mock()
        equity_values = np.random.rand(100) * 10000 + 100000
        portfolio.value.return_value = equity_values

        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        equity = futures_pf.get_equity_curve()

        # Assert: Shape should match
        assert equity.shape == equity_values.shape
        assert len(equity) == 100


class TestFormatPrice:
    """Tests for tick size rounding."""

    @pytest.mark.parametrize(
        "price,tick_size,expected",
        [
            (4500.13, 0.25, 4500.25),  # Round up to nearest 0.25
            (4500.12, 0.25, 4500.00),  # Round down to nearest 0.25
            (100.567, 0.01, 100.57),  # Round up to nearest 0.01
            (100.563, 0.01, 100.56),  # Round down to nearest 0.01
            (1234.6, 1.0, 1235.0),  # Round up to nearest 1.0
            (1234.4, 1.0, 1234.0),  # Round down to nearest 1.0
            (4500.00, 0.25, 4500.00),  # Already on tick
            (100.50, 0.01, 100.50),  # Already on tick
        ],
    )
    def test_format_price_rounds_to_tick(
        self, price: float, tick_size: float, expected: float
    ) -> None:
        """format_price() should round to nearest tick_size."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, tick_size)

        # Act
        formatted = futures_pf.format_price(price)

        # Assert
        assert abs(formatted - expected) < 1e-10

    def test_format_price_does_not_affect_calculations(self) -> None:
        """Tick size should be for display only, not affecting PnL calculations."""
        # Arrange: Two portfolios with same profit but different tick sizes
        portfolio = mock.Mock()
        portfolio.total_profit.return_value = 100.0
        portfolio.total_return.return_value = 0.10
        portfolio.init_cash = 100000.0

        trades = mock.Mock()
        closed = mock.Mock()
        closed.count.return_value = 0
        trades.closed = closed
        portfolio.trades = trades

        returns_acc = mock.Mock()
        returns_acc.sharpe_ratio.return_value = 1.0
        returns_acc.sortino_ratio.return_value = 1.0
        returns_acc.calmar_ratio.return_value = 1.0
        portfolio.returns_acc = returns_acc

        drawdowns = mock.Mock()
        drawdowns.max_drawdown.return_value = -0.15
        portfolio.drawdowns = drawdowns

        futures_pf_1 = FuturesPortfolio(portfolio, 50.0, 0.25)
        futures_pf_2 = FuturesPortfolio(portfolio, 50.0, 1.0)

        # Act
        analytics_1 = futures_pf_1.get_analytics()
        analytics_2 = futures_pf_2.get_analytics()

        # Assert: PnL should be identical regardless of tick_size
        assert abs(analytics_1.total_pnl - analytics_2.total_pnl) < 1e-10
        assert analytics_1.tick_size == 0.25
        assert analytics_2.tick_size == 1.0


class TestSafeFloat:
    """Tests for _safe_float helper method."""

    def test_safe_float_with_none(self) -> None:
        """_safe_float should convert None to default (0.0)."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result = futures_pf._safe_float(None)

        # Assert
        assert result == 0.0

    def test_safe_float_with_nan(self) -> None:
        """_safe_float should convert NaN to default (0.0)."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result = futures_pf._safe_float(float("nan"))

        # Assert
        assert result == 0.0

    def test_safe_float_with_inf(self) -> None:
        """_safe_float should convert inf to default (0.0)."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result_pos_inf = futures_pf._safe_float(float("inf"))
        result_neg_inf = futures_pf._safe_float(float("-inf"))

        # Assert
        assert result_pos_inf == 0.0
        assert result_neg_inf == 0.0

    def test_safe_float_with_valid_number(self) -> None:
        """_safe_float should return valid float unchanged."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result = futures_pf._safe_float(123.45)

        # Assert
        assert abs(result - 123.45) < 1e-10

    def test_safe_float_with_custom_default(self) -> None:
        """_safe_float should use custom default when provided."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result = futures_pf._safe_float(None, default=99.0)

        # Assert
        assert result == 99.0

    def test_safe_float_with_string_returns_default(self) -> None:
        """_safe_float should return default for unconvertible types."""
        # Arrange
        portfolio = mock.Mock()
        futures_pf = FuturesPortfolio(portfolio, 50.0, 0.25)

        # Act
        result = futures_pf._safe_float("not a number", default=42.0)

        # Assert
        assert result == 42.0


class TestPortfolioAnalyticsDataclass:
    """Tests for PortfolioAnalytics dataclass structure."""

    def test_dataclass_fields_exist(self) -> None:
        """PortfolioAnalytics should have all expected fields."""
        # Arrange & Act: Create instance
        analytics = PortfolioAnalytics(
            total_pnl=5000.0,
            avg_trade_pnl=50.0,
            avg_win_dollars=100.0,
            avg_loss_dollars=50.0,
            max_win_dollars=200.0,
            max_loss_dollars=100.0,
            max_drawdown_dollars=10000.0,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=0.5,
            profit_factor=2.0,
            total_return=0.05,
            max_drawdown_percent=0.10,
            win_rate=0.60,
            n_trades=100,
            n_wins=60,
            n_losses=40,
            point_value=50.0,
            tick_size=0.25,
            total_fees=250.0,
            expectancy=50.0,
        )

        # Assert: All fields should be accessible
        assert analytics.total_pnl == 5000.0
        assert analytics.avg_trade_pnl == 50.0
        assert analytics.avg_win_dollars == 100.0
        assert analytics.avg_loss_dollars == 50.0
        assert analytics.max_win_dollars == 200.0
        assert analytics.max_loss_dollars == 100.0
        assert analytics.max_drawdown_dollars == 10000.0
        assert analytics.sharpe_ratio == 1.5
        assert analytics.sortino_ratio == 1.8
        assert analytics.calmar_ratio == 0.5
        assert analytics.profit_factor == 2.0
        assert analytics.total_return == 0.05
        assert analytics.max_drawdown_percent == 0.10
        assert analytics.win_rate == 0.60
        assert analytics.n_trades == 100
        assert analytics.n_wins == 60
        assert analytics.n_losses == 40
        assert analytics.point_value == 50.0
        assert analytics.tick_size == 0.25
        assert analytics.total_fees == 250.0
        assert analytics.expectancy == 50.0
