"""
Backtesting Engine
Core backtesting framework for evaluating trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
from .performance import PerformanceMetrics
from .risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    direction: str  # 'long' or 'short'
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    
    def close_trade(self, exit_date: datetime, exit_price: float):
        """Close the trade and calculate P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.direction == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.return_pct = (self.entry_price - exit_price) / self.entry_price


class BacktestEngine:
    """
    Main backtesting engine for strategy evaluation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.2
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            position_size: Maximum position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        
        # State variables
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
    def run(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        **strategy_params
    ) -> 'BacktestResults':
        """
        Run backtest for a given strategy
        
        Args:
            strategy: Strategy function that returns signals
            data: Price data DataFrame
            symbol: Stock symbol
            **strategy_params: Additional parameters for strategy
        
        Returns:
            BacktestResults object
        """
        logger.info(f"Starting backtest for {symbol} from {data.index[0]} to {data.index[-1]}")
        
        # Reset state
        self._reset()
        
        # Generate signals
        signals = strategy(data, **strategy_params)
        
        if isinstance(signals, pd.Series):
            signals = signals.to_frame('signal')
        
        # Merge signals with data
        data_with_signals = data.copy()
        data_with_signals['signal'] = signals['signal']
        
        # Execute trades based on signals
        for idx, row in data_with_signals.iterrows():
            self._process_bar(idx, row)
        
        # Close any open positions
        if self.current_position:
            last_row = data_with_signals.iloc[-1]
            self._close_position(data_with_signals.index[-1], last_row['Close'])
        
        # Calculate results
        results = self._calculate_results(data_with_signals, symbol)
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return results
    
    def _reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_position = None
    
    def _process_bar(self, date: datetime, bar: pd.Series):
        """Process a single bar of data"""
        signal = bar.get('signal', 0)
        close_price = bar['Close']
        
        # Check for exit signal
        if self.current_position and signal == 0:
            self._close_position(date, close_price)
        
        # Check for entry signal
        elif not self.current_position:
            if signal == 1:  # Long signal
                self._open_position(date, close_price, 'long')
            elif signal == -1:  # Short signal
                self._open_position(date, close_price, 'short')
        
        # Update equity curve
        current_equity = self._calculate_current_equity(close_price)
        self.equity_curve.append(current_equity)
    
    def _open_position(self, date: datetime, price: float, direction: str):
        """Open a new position"""
        # Apply slippage
        if direction == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)
        
        # Calculate position size
        position_value = self.capital * self.position_size
        quantity = int(position_value / entry_price)
        
        if quantity == 0:
            return
        
        # Apply commission
        commission_cost = entry_price * quantity * self.commission
        
        # Create trade
        trade = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            direction=direction
        )
        
        self.current_position = trade
        self.capital -= commission_cost
        
        logger.debug(f"Opened {direction} position: {quantity} @ {entry_price:.2f}")
    
    def _close_position(self, date: datetime, price: float):
        """Close current position"""
        if not self.current_position:
            return
        
        # Apply slippage
        if self.current_position.direction == 'long':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        
        # Close trade
        self.current_position.close_trade(date, exit_price)
        
        # Apply commission
        commission_cost = exit_price * self.current_position.quantity * self.commission
        
        # Update capital
        self.capital += self.current_position.pnl - commission_cost
        
        # Store trade
        self.trades.append(self.current_position)
        
        logger.debug(f"Closed position: P&L = {self.current_position.pnl:.2f}")
        
        self.current_position = None
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        if self.current_position:
            # Mark-to-market
            if self.current_position.direction == 'long':
                unrealized_pnl = (current_price - self.current_position.entry_price) * self.current_position.quantity
            else:
                unrealized_pnl = (self.current_position.entry_price - current_price) * self.current_position.quantity
            
            return self.capital + unrealized_pnl
        else:
            return self.capital
    
    def _calculate_results(self, data: pd.DataFrame, symbol: Optional[str]) -> 'BacktestResults':
        """Calculate backtest results"""
        results = BacktestResults(
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=self.equity_curve,
            benchmark_data=data
        )
        
        return results


class BacktestResults:
    """
    Container for backtest results and performance metrics
    """
    
    def __init__(
        self,
        symbol: Optional[str],
        initial_capital: float,
        final_capital: float,
        trades: List[Trade],
        equity_curve: List[float],
        benchmark_data: pd.DataFrame
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.trades = trades
        self.equity_curve = equity_curve
        self.benchmark_data = benchmark_data
        
        # Calculate metrics
        self.performance = PerformanceMetrics(self)
        self.risk = RiskMetrics(self)
    
    def get_metrics(self) -> Dict:
        """Get all performance metrics"""
        return {
            **self.performance.calculate_all(),
            **self.risk.calculate_all()
        }
    
    def plot_performance(self):
        """Plot equity curve and drawdown"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        ax1.plot(self.equity_curve, label='Strategy')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.set_title(f'Equity Curve - {self.symbol}')
        ax1.set_ylabel('Capital')
        ax1.legend()
        ax1.grid(True)
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print summary of backtest results"""
        metrics = self.get_metrics()
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS - {self.symbol}")
        print(f"{'='*60}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  CAGR: {metrics['cagr']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"\nRISK METRICS:")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Volatility: {metrics['volatility']:.2f}%")
        print(f"  VaR (95%): {metrics['var_95']:.2f}%")
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Trade Return: {metrics['avg_trade_return']:.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    from data.data_pipeline import DataPipeline
    
    # Fetch data
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01', end_date='2023-12-31')
    
    # Simple moving average crossover strategy
    def sma_crossover_strategy(data, short_window=20, long_window=50):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate moving averages
        signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
        signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals['signal'][short_window:] = np.where(
            signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1, 0
        )
        
        # Generate trading orders
        signals['positions'] = signals['signal'].diff()
        
        return signals['signal']
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(sma_crossover_strategy, data, symbol='TCS.NS')
    
    # Print results
    results.print_summary()

