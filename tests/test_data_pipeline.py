"""
Unit tests for data pipeline
"""

import pytest
import pandas as pd
from data.data_pipeline import DataPipeline


def test_data_pipeline_init():
    """Test data pipeline initialization"""
    pipeline = DataPipeline()
    assert pipeline is not None


def test_fetch_stock_data():
    """Test fetching stock data"""
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2024-01-01', end_date='2024-01-31')
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'Close' in data.columns


def test_invalid_symbol():
    """Test handling of invalid symbol"""
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('INVALID_SYMBOL_XYZ', start_date='2024-01-01')
    
    assert isinstance(data, pd.DataFrame)
    # Should return empty DataFrame for invalid symbol

