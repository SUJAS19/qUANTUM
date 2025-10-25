"""
Mean Reversion Strategy
Trade when prices deviate from their mean and revert back
"""

import pandas as pd
import numpy as np
from typing import Optional


class MeanReversionStrategy:
    """
    Bollinger Bands based mean reversion strategy
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, rsi_period: int = 14):
        """
        Initialize mean reversion strategy
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations for Bollinger Bands
            rsi_period: RSI period
        """
        self.window = window
        self.num_std = num_std
        self.rsi_period = rsi_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Returns:
            Series with signals: 1 (buy), 0 (hold), -1 (sell)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=self.window).mean()
        std = data['Close'].rolling(window=self.window).std()
        
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        # Buy when price touches lower band and RSI < 30
        buy_signal = (data['Close'] <= lower_band) & (rsi < 30)
        
        # Sell when price touches upper band and RSI > 70
        sell_signal = (data['Close'] >= upper_band) & (rsi > 70)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
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
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01')
    
    # Create strategy
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(strategy, data, symbol='TCS.NS')
    
    results.print_summary()
    results.plot_performance()

