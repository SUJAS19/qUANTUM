"""
Technical Analysis Module
Calculate technical indicators for trading analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Technical analysis indicators and tools
    """
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            num_std: Number of standard deviations
        
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
        
        Returns:
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
        
        Returns:
            ADX values
        """
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close: Close prices
            volume: Volume
        
        Returns:
            OBV values
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
        
        Returns:
            VWAP values
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    def identify_support_resistance(
        prices: pd.Series,
        window: int = 20,
        num_levels: int = 3
    ) -> Tuple[list, list]:
        """
        Identify support and resistance levels
        
        Args:
            prices: Price series
            window: Window for local extrema
            num_levels: Number of levels to identify
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find local minima (support)
        local_min = prices[(prices.shift(1) > prices) & (prices.shift(-1) > prices)]
        
        # Find local maxima (resistance)
        local_max = prices[(prices.shift(1) < prices) & (prices.shift(-1) < prices)]
        
        # Get top support levels
        support_levels = sorted(local_min.values)[:num_levels] if len(local_min) > 0 else []
        
        # Get top resistance levels
        resistance_levels = sorted(local_max.values, reverse=True)[:num_levels] if len(local_max) > 0 else []
        
        return support_levels, resistance_levels
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple technical indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=df.index)
        
        # Calculate indicators
        signals['rsi'] = TechnicalAnalyzer.calculate_rsi(df['Close'])
        macd, signal, hist = TechnicalAnalyzer.calculate_macd(df['Close'])
        signals['macd'] = macd
        signals['macd_signal'] = signal
        signals['macd_hist'] = hist
        
        # Calculate moving averages
        signals['sma_20'] = df['Close'].rolling(window=20).mean()
        signals['sma_50'] = df['Close'].rolling(window=50).mean()
        signals['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        signals['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Generate buy/sell signals
        signals['signal'] = 0
        
        # RSI signals
        signals.loc[signals['rsi'] < 30, 'signal'] += 1  # Oversold
        signals.loc[signals['rsi'] > 70, 'signal'] -= 1  # Overbought
        
        # MACD signals
        signals.loc[signals['macd'] > signals['macd_signal'], 'signal'] += 1
        signals.loc[signals['macd'] < signals['macd_signal'], 'signal'] -= 1
        
        # MA crossover signals
        signals.loc[signals['sma_20'] > signals['sma_50'], 'signal'] += 1
        signals.loc[signals['sma_20'] < signals['sma_50'], 'signal'] -= 1
        
        # Normalize signals to -1, 0, 1
        signals['signal'] = signals['signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        return signals


if __name__ == "__main__":
    # Example usage
    from data.data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
    
    analyzer = TechnicalAnalyzer()
    
    # Calculate RSI
    rsi = analyzer.calculate_rsi(data['Close'])
    print("RSI:")
    print(rsi.tail())
    
    # Calculate MACD
    macd, signal, hist = analyzer.calculate_macd(data['Close'])
    print("\nMACD:")
    print(f"MACD: {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}")
    
    # Generate signals
    signals = analyzer.generate_signals(data)
    print("\nLatest Signal:")
    print(signals.tail())

