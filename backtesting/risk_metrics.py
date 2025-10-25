"""
Risk Metrics Module
Calculate various risk metrics for backtesting
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest_engine import BacktestResults


class RiskMetrics:
    """
    Calculate risk metrics for backtest results
    """
    
    def __init__(self, results: 'BacktestResults'):
        self.results = results
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown percentage
        
        Returns:
            Maximum drawdown as percentage
        """
        if len(self.results.equity_curve) == 0:
            return 0.0
        
        equity_series = pd.Series(self.results.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        return drawdown.min()
    
    def volatility(self, annualized: bool = True) -> float:
        """
        Calculate volatility of returns
        
        Args:
            annualized: If True, return annualized volatility
        
        Returns:
            Volatility as percentage
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        
        if annualized:
            vol = vol * np.sqrt(252)  # Assuming 252 trading days
        
        return vol * 100
    
    def downside_deviation(self, annualized: bool = True) -> float:
        """
        Calculate downside deviation (only negative returns)
        
        Args:
            annualized: If True, return annualized deviation
        
        Returns:
            Downside deviation as percentage
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        dd = downside_returns.std()
        
        if annualized:
            dd = dd * np.sqrt(252)
        
        return dd * 100
    
    def var_95(self) -> float:
        """
        Calculate Value at Risk at 95% confidence level
        
        Returns:
            VaR as percentage
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, 5) * 100
    
    def cvar_95(self) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall) at 95% confidence
        
        Returns:
            CVaR as percentage
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        var_threshold = np.percentile(returns, 5)
        cvar = returns[returns <= var_threshold].mean()
        
        return cvar * 100
    
    def max_consecutive_losses(self) -> int:
        """Calculate maximum number of consecutive losing trades"""
        if len(self.results.trades) == 0:
            return 0
        
        max_losses = 0
        current_losses = 0
        
        for trade in self.results.trades:
            if trade.pnl and trade.pnl < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def max_consecutive_wins(self) -> int:
        """Calculate maximum number of consecutive winning trades"""
        if len(self.results.trades) == 0:
            return 0
        
        max_wins = 0
        current_wins = 0
        
        for trade in self.results.trades:
            if trade.pnl and trade.pnl > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    def recovery_factor(self) -> float:
        """
        Calculate recovery factor (Net Profit / Max Drawdown)
        
        Returns:
            Recovery factor
        """
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        net_profit = self.results.final_capital - self.results.initial_capital
        return (net_profit / self.results.initial_capital * 100) / max_dd
    
    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (measure of downside risk)
        
        Returns:
            Ulcer index
        """
        if len(self.results.equity_curve) == 0:
            return 0.0
        
        equity_series = pd.Series(self.results.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown_pct = ((equity_series - running_max) / running_max * 100) ** 2
        
        return np.sqrt(drawdown_pct.mean())
    
    def calculate_all(self) -> dict:
        """Calculate all risk metrics"""
        return {
            'max_drawdown': self.max_drawdown(),
            'volatility': self.volatility(),
            'downside_deviation': self.downside_deviation(),
            'var_95': self.var_95(),
            'cvar_95': self.cvar_95(),
            'max_consecutive_losses': self.max_consecutive_losses(),
            'max_consecutive_wins': self.max_consecutive_wins(),
            'recovery_factor': self.recovery_factor(),
            'ulcer_index': self.ulcer_index()
        }

