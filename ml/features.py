"""
Feature Engineering Module
Create technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create features from price data for machine learning
    """
    
    @staticmethod
    def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Add return features for different periods"""
        for period in periods:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add moving average features"""
        for window in windows:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            
            # Price relative to MA
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}'] - 1
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """Add MACD indicators"""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        
        df['bb_upper'] = sma + (std * num_std)
        df['bb_lower'] = sma - (std * num_std)
        df['bb_middle'] = sma
        
        # Bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        df['atr'] = true_range.rolling(window=period).mean()
        df['atr_pct'] = df['atr'] / df['Close']
        
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Volume price trend
        df['vpt'] = (df['Volume'] * ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))).cumsum()
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['Close'].pct_change(period) * 100
        
        # Momentum
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        return df
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # Historical volatility
        returns = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        # Higher highs, lower lows
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        # Gap up/down
        df['gap_up'] = ((df['Open'] > df['High'].shift(1))).astype(int)
        df['gap_down'] = ((df['Open'] < df['Low'].shift(1))).astype(int)
        
        # Candle body and shadow
        df['body'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        df['body_pct'] = df['body'] / df['Close']
        
        return df
    
    @staticmethod
    def add_lagged_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged price features"""
        for lag in lags:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    @staticmethod
    def add_statistical_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        df[f'mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'max_{window}'] = df['Close'].rolling(window=window).max()
        
        # Position in range
        df[f'pct_in_range_{window}'] = (df['Close'] - df[f'min_{window}']) / (df[f'max_{window}'] - df[f'min_{window}'])
        
        # Z-score
        df[f'zscore_{window}'] = (df['Close'] - df[f'mean_{window}']) / df[f'std_{window}']
        
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Is month/quarter end
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    @staticmethod
    def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend identification features"""
        # ADX (Average Directional Index)
        period = 14
        
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        
        # Trend strength
        df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features at once
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        logger.info("Creating features...")
        
        # Add all feature groups
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_stochastic(df)
        df = self.add_volume_features(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.add_price_patterns(df)
        df = self.add_lagged_features(df)
        df = self.add_statistical_features(df)
        df = self.add_time_features(df)
        df = self.add_trend_features(df)
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def get_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str = 'Close',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance using Random Forest
        
        Args:
            df: DataFrame with features
            target_col: Target column
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        from sklearn.ensemble import RandomForestRegressor
        
        features_df = self.create_all_features(df).dropna()
        
        exclude_cols = [target_col, 'Date'] if 'Date' in features_df.columns else [target_col]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == "__main__":
    # Example usage
    from data.data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
    
    engineer = FeatureEngineer()
    
    # Create all features
    features_df = engineer.create_all_features(data)
    
    print("Features created:")
    print(f"Total features: {len(features_df.columns)}")
    print("\nSample features:")
    print(features_df.head())
    
    # Get feature importance
    importance = engineer.get_feature_importance(data, top_n=10)
    
    print("\nTop 10 Important Features:")
    print(importance)

