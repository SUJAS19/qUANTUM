"""
Pairs Trading Strategy
Statistical arbitrage between correlated stocks
"""

import pandas as pd
import numpy as np
from typing import Tuple


class PairsTradingStrategy:
    """
    Pairs trading based on cointegration and z-score
    """
    
    def __init__(self, window: int = 30, entry_zscore: float = 2.0, exit_zscore: float = 0.5):
        """
        Initialize pairs trading strategy
        
        Args:
            window: Rolling window for mean and std calculation
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
        """
        self.window = window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
    
    def calculate_spread(self, stock1: pd.Series, stock2: pd.Series) -> pd.Series:
        """Calculate spread between two stocks"""
        # Simple spread (can be improved with cointegration)
        return stock1 - stock2
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of spread"""
        mean = spread.rolling(window=self.window).mean()
        std = spread.rolling(window=self.window).std()
        
        zscore = (spread - mean) / std
        
        return zscore
    
    def generate_signals(
        self,
        stock1_data: pd.DataFrame,
        stock2_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals for both stocks
        
        Returns:
            Tuple of (signals_stock1, signals_stock2)
        """
        # Calculate spread
        spread = self.calculate_spread(stock1_data['Close'], stock2_data['Close'])
        
        # Calculate z-score
        zscore = self.calculate_zscore(spread)
        
        # Initialize signals
        signals_stock1 = pd.Series(0, index=stock1_data.index)
        signals_stock2 = pd.Series(0, index=stock2_data.index)
        
        # Generate signals
        # When z-score > entry_threshold: short stock1, long stock2
        short_signal = zscore > self.entry_zscore
        signals_stock1[short_signal] = -1
        signals_stock2[short_signal] = 1
        
        # When z-score < -entry_threshold: long stock1, short stock2
        long_signal = zscore < -self.entry_zscore
        signals_stock1[long_signal] = 1
        signals_stock2[long_signal] = -1
        
        # Exit when z-score returns to near zero
        exit_signal = abs(zscore) < self.exit_zscore
        signals_stock1[exit_signal] = 0
        signals_stock2[exit_signal] = 0
        
        return signals_stock1, signals_stock2
    
    @staticmethod
    def find_cointegrated_pairs(data_dict: dict, significance_level: float = 0.05) -> list:
        """
        Find cointegrated pairs from a dictionary of stock data
        
        Args:
            data_dict: Dictionary with symbol as key and DataFrame as value
            significance_level: P-value threshold for cointegration
        
        Returns:
            List of cointegrated pairs
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            symbols = list(data_dict.keys())
            pairs = []
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    stock1 = symbols[i]
                    stock2 = symbols[j]
                    
                    # Get price data
                    prices1 = data_dict[stock1]['Close']
                    prices2 = data_dict[stock2]['Close']
                    
                    # Align data
                    aligned = pd.concat([prices1, prices2], axis=1).dropna()
                    
                    if len(aligned) < 30:
                        continue
                    
                    # Test for cointegration
                    score, pvalue, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    
                    if pvalue < significance_level:
                        pairs.append({
                            'stock1': stock1,
                            'stock2': stock2,
                            'pvalue': pvalue,
                            'cointegration_score': score
                        })
            
            return sorted(pairs, key=lambda x: x['pvalue'])
        
        except ImportError:
            print("statsmodels not installed, cannot test for cointegration")
            return []


if __name__ == "__main__":
    from data.data_pipeline import DataPipeline
    
    # Fetch data for two correlated stocks
    pipeline = DataPipeline()
    
    stock1_data = pipeline.fetch_stock_data('HDFCBANK.NS', start_date='2023-01-01')
    stock2_data = pipeline.fetch_stock_data('ICICIBANK.NS', start_date='2023-01-01')
    
    # Create strategy
    strategy = PairsTradingStrategy(window=30, entry_zscore=2.0)
    
    # Generate signals
    signals1, signals2 = strategy.generate_signals(stock1_data, stock2_data)
    
    print("Pairs Trading Signals:")
    print(f"HDFC Bank signals: {signals1[signals1 != 0].count()}")
    print(f"ICICI Bank signals: {signals2[signals2 != 0].count()}")
    
    # Find cointegrated pairs
    data_dict = {
        'HDFCBANK.NS': stock1_data,
        'ICICIBANK.NS': stock2_data
    }
    
    pairs = PairsTradingStrategy.find_cointegrated_pairs(data_dict)
    
    if pairs:
        print("\nCointegrated Pairs:")
        for pair in pairs:
            print(f"{pair['stock1']} - {pair['stock2']}: p-value = {pair['pvalue']:.4f}")

