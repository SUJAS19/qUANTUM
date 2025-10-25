"""
Performance Metrics Module
Calculate various performance metrics for backtesting
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest_engine import BacktestResults


class PerformanceMetrics:
    """
    Calculate performance metrics for backtest results
    """
    
    def __init__(self, results: 'BacktestResults'):
        self.results = results
    
    def total_return(self) -> float:
        """Calculate total return percentage"""
        return ((self.results.final_capital - self.results.initial_capital) / 
                self.results.initial_capital * 100)
    
    def cagr(self) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(self.results.equity_curve) == 0:
            return 0.0
        
        years = len(self.results.equity_curve) / 252  # Assuming 252 trading days
        
        if years == 0:
            return 0.0
        
        return ((self.results.final_capital / self.results.initial_capital) ** (1 / years) - 1) * 100
    
    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sortino Ratio (only considers downside volatility)
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        if len(self.results.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.results.equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio (CAGR / Max Drawdown)
        
        Returns:
            Calmar ratio
        """
        from .risk_metrics import RiskMetrics
        
        max_dd = RiskMetrics(self.results).max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return self.cagr() / abs(max_dd)
    
    def total_trades(self) -> int:
        """Total number of trades"""
        return len(self.results.trades)
    
    def win_rate(self) -> float:
        """Percentage of winning trades"""
        if len(self.results.trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in self.results.trades if trade.pnl and trade.pnl > 0)
        return (winning_trades / len(self.results.trades)) * 100
    
    def profit_factor(self) -> float:
        """
        Profit factor (gross profits / gross losses)
        
        Returns:
            Profit factor
        """
        if len(self.results.trades) == 0:
            return 0.0
        
        gross_profit = sum(trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def avg_trade_return(self) -> float:
        """Average return per trade (percentage)"""
        if len(self.results.trades) == 0:
            return 0.0
        
        returns = [trade.return_pct for trade in self.results.trades if trade.return_pct is not None]
        
        if len(returns) == 0:
            return 0.0
        
        return np.mean(returns) * 100
    
    def avg_win(self) -> float:
        """Average winning trade"""
        wins = [trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl > 0]
        return np.mean(wins) if wins else 0.0
    
    def avg_loss(self) -> float:
        """Average losing trade"""
        losses = [trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl < 0]
        return np.mean(losses) if losses else 0.0
    
    def largest_win(self) -> float:
        """Largest winning trade"""
        wins = [trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl > 0]
        return max(wins) if wins else 0.0
    
    def largest_loss(self) -> float:
        """Largest losing trade"""
        losses = [trade.pnl for trade in self.results.trades if trade.pnl and trade.pnl < 0]
        return min(losses) if losses else 0.0
    
    def calculate_all(self) -> dict:
        """Calculate all performance metrics"""
        return {
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'total_trades': self.total_trades(),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_trade_return': self.avg_trade_return(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'largest_win': self.largest_win(),
            'largest_loss': self.largest_loss()
        }

