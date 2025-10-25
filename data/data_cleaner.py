"""
Data Cleaning and Validation Module
Handles data quality checks and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaning and validation utilities
    """
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on date index"""
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate dates")
            df = df[~df.index.duplicated(keep='first')]
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate', or 'drop'
        
        Returns:
            Cleaned DataFrame
        """
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            
            if method == 'ffill':
                df = df.fillna(method='ffill')
            elif method == 'bfill':
                df = df.fillna(method='bfill')
            elif method == 'interpolate':
                df = df.interpolate(method='linear')
            elif method == 'drop':
                df = df.dropna()
            else:
                logger.warning(f"Unknown method {method}, using forward fill")
                df = df.fillna(method='ffill')
        
        return df
    
    @staticmethod
    def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC data (High >= Low, High >= Open/Close, etc.)
        
        Args:
            df: DataFrame with OHLC columns
        
        Returns:
            Validated DataFrame with invalid rows removed or corrected
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            logger.warning("Missing OHLC columns, skipping validation")
            return df
        
        # Check for logical inconsistencies
        invalid_high_low = df['High'] < df['Low']
        invalid_high_open = df['High'] < df['Open']
        invalid_high_close = df['High'] < df['Close']
        invalid_low_open = df['Low'] > df['Open']
        invalid_low_close = df['Low'] > df['Close']
        
        invalid_mask = (
            invalid_high_low | 
            invalid_high_open | 
            invalid_high_close | 
            invalid_low_open | 
            invalid_low_close
        )
        
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} rows with invalid OHLC data")
            df = df[~invalid_mask]
        
        return df
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame, 
        column: str, 
        method: str = 'iqr', 
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from a specific column
        
        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            logger.warning(f"Unknown method {method}")
            return df
        
        if outliers.any():
            logger.info(f"Removed {outliers.sum()} outliers from {column}")
            df = df[~outliers]
        
        return df
    
    @staticmethod
    def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Add return columns for different periods
        
        Args:
            df: DataFrame with 'Close' column
            periods: List of periods for return calculation
        
        Returns:
            DataFrame with return columns added
        """
        if 'Close' not in df.columns:
            logger.warning("Close column not found")
            return df
        
        for period in periods:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        if 'Volume' not in df.columns:
            logger.warning("Volume column not found")
            return df
        
        # Average volume (20-day)
        df['Avg_Volume_20d'] = df['Volume'].rolling(window=20).mean()
        
        # Volume ratio (current vs average)
        df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume_20d']
        
        return df
    
    @staticmethod
    def resample_data(
        df: pd.DataFrame, 
        timeframe: str = 'W', 
        agg_dict: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            df: Input DataFrame with datetime index
            timeframe: 'D' (daily), 'W' (weekly), 'M' (monthly)
            agg_dict: Custom aggregation dictionary
        
        Returns:
            Resampled DataFrame
        """
        if agg_dict is None:
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
        
        # Filter only existing columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        if not agg_dict:
            logger.warning("No valid columns for resampling")
            return df
        
        resampled = df.resample(timeframe).agg(agg_dict)
        return resampled.dropna()
    
    @staticmethod
    def clean_pipeline(df: pd.DataFrame, full_clean: bool = True) -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        
        Args:
            df: Input DataFrame
            full_clean: If True, perform all cleaning steps
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline")
        
        # Remove duplicates
        df = DataCleaner.remove_duplicates(df)
        
        # Validate OHLC
        df = DataCleaner.validate_ohlc(df)
        
        # Handle missing values
        df = DataCleaner.handle_missing_values(df, method='ffill')
        
        if full_clean:
            # Add returns
            df = DataCleaner.add_returns(df)
            
            # Add volume indicators
            df = DataCleaner.add_volume_indicators(df)
        
        logger.info("Data cleaning complete")
        return df


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    df = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01')
    
    print("Before cleaning:")
    print(df.info())
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_pipeline(df)
    
    print("\nAfter cleaning:")
    print(df_clean.info())
    print(df_clean.tail())

