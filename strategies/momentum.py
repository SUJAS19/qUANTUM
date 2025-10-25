"""
Momentum Strategy
Trade based on price momentum and trend
"""

import pandas as pd
import numpy as np


class MomentumStrategy:
    """
    Moving Average Crossover and RSI momentum strategy
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50, rsi_period: int = 14):
        """
        Initialize momentum strategy
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
            rsi_period: RSI period
        """
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Returns:
            Series with signals: 1 (buy), 0 (hold), -1 (sell)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        # Buy when short MA crosses above long MA and RSI > 50
        buy_signal = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)) & (rsi > 50)
        
        # Sell when short MA crosses below long MA
        sell_signal = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        # Maintain position
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        # Forward fill to maintain positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
    
    def __call__(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Make strategy callable for backtesting"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = self.generate_signals(data)
        return signals


if __name__ == "__main__":
    from data.data_pipeline import DataPipeline
    from backtesting.backtest_engine import BacktestEngine
    
    # Fetch data
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
    
    # Create strategy
    strategy = MomentumStrategy(short_window=20, long_window=50)
    
    # Backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(strategy, data, symbol='RELIANCE.NS')
    
    results.print_summary()

