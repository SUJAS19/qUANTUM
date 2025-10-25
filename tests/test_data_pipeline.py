"""
Unit tests for data pipeline
"""

import pytest
import pandas as pd
import numpy as np
from data.data_pipeline import DataPipeline
from data.data_cleaner import DataCleaner


def test_data_pipeline_init():
    """Test data pipeline initialization"""
    pipeline = DataPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'raw_data_path')
    assert hasattr(pipeline, 'processed_data_path')


def test_data_cleaner_remove_duplicates():
    """Test duplicate removal"""
    # Create sample data with duplicates
    data = pd.DataFrame({
        'Close': [100, 101, 102, 102],
        'Volume': [1000, 1100, 1200, 1200]
    }, index=pd.date_range('2024-01-01', periods=4))
    
    # Add duplicate
    data = pd.concat([data, data.iloc[[-1]]])
    
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(data)
    
    assert len(cleaned) == 4  # One duplicate should be removed


def test_data_cleaner_handle_missing():
    """Test missing value handling"""
    # Create data with missing values
    data = pd.DataFrame({
        'Close': [100, np.nan, 102, 103],
        'Volume': [1000, 1100, np.nan, 1300]
    })
    
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(data, method='ffill')
    
    assert not cleaned.isnull().any().any()


def test_data_cleaner_validate_ohlc():
    """Test OHLC validation"""
    # Create valid OHLC data
    data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [104, 105, 106]
    })
    
    cleaner = DataCleaner()
    validated = cleaner.validate_ohlc(data)
    
    assert len(validated) == 3  # All rows should be valid


def test_config_loading():
    """Test configuration loading"""
    pipeline = DataPipeline()
    assert pipeline.config is not None
    assert 'data' in pipeline.config

