"""
Data Pipeline for automated market data collection
Handles OHLCV data and derivatives data for Indian equities
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import yaml
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Automated data pipeline for fetching and storing market data
    """
    
    def __init__(self, config_path: str = "config/config.example.yaml"):
        """Initialize data pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.symbols_config = self._load_symbols()
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _load_symbols(self) -> Dict:
        """Load symbols from YAML file"""
        try:
            with open('config/symbols.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Symbols file not found, using empty list")
            return {'nifty50': []}
    
    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'data': {
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed',
                'default_start_date': '2020-01-01',
                'default_end_date': datetime.now().strftime('%Y-%m-%d')
            }
        }
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = self.config['data']['default_start_date']
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Save to raw data
            self._save_raw_data(df, symbol)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_nifty50_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all NIFTY 50 stocks
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        nifty50_symbols = self.symbols_config.get('nifty50', [])
        
        if not nifty50_symbols:
            logger.error("No NIFTY 50 symbols found in config")
            return {}
        
        data_dict = {}
        
        logger.info(f"Fetching data for {len(nifty50_symbols)} NIFTY 50 stocks")
        
        for symbol in tqdm(nifty50_symbols, desc="Fetching NIFTY 50 data"):
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if not df.empty:
                data_dict[symbol] = df
        
        # Save combined data
        self._save_combined_data(data_dict, 'nifty50_data.csv')
        
        logger.info(f"Successfully fetched data for {len(data_dict)} stocks")
        return data_dict
    
    def fetch_multiple_stocks(
        self, 
        symbols: List[str], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in tqdm(symbols, desc="Fetching stock data"):
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if not df.empty:
                data_dict[symbol] = df
        
        return data_dict
    
    def fetch_index_data(
        self, 
        index_symbol: str = "^NSEI", 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch index data (NIFTY, BANK NIFTY, etc.)
        
        Args:
            index_symbol: Index symbol (^NSEI for NIFTY 50)
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with index data
        """
        return self.fetch_stock_data(index_symbol, start_date, end_date)
    
    def fetch_option_chain(self, symbol: str = "NIFTY", expiry_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch option chain data for a symbol
        
        Note: This is a placeholder. Actual implementation would require
        NSE API or scraping (which has limitations)
        
        Args:
            symbol: Underlying symbol
            expiry_date: Option expiry date
        
        Returns:
            DataFrame with option chain data
        """
        logger.warning("Option chain fetching requires NSE API access or web scraping")
        logger.info("For now, returning sample structure")
        
        # Sample structure for option chain
        option_chain = pd.DataFrame({
            'Strike': [],
            'Call_OI': [],
            'Call_Volume': [],
            'Call_IV': [],
            'Call_LTP': [],
            'Put_LTP': [],
            'Put_IV': [],
            'Put_Volume': [],
            'Put_OI': []
        })
        
        return option_chain
    
    def calculate_greeks(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega)
        
        Note: Requires option pricing model implementation
        """
        logger.warning("Greeks calculation requires option pricing model")
        return option_data
    
    def _save_raw_data(self, df: pd.DataFrame, symbol: str):
        """Save raw data to CSV"""
        filename = f"{symbol.replace('.NS', '')}_raw.csv"
        filepath = self.raw_data_path / filename
        df.to_csv(filepath)
        logger.debug(f"Saved raw data to {filepath}")
    
    def _save_combined_data(self, data_dict: Dict[str, pd.DataFrame], filename: str):
        """Save combined data from multiple stocks"""
        if not data_dict:
            return
        
        combined_df = pd.concat(data_dict.values(), ignore_index=True)
        filepath = self.processed_data_path / filename
        combined_df.to_csv(filepath, index=False)
        logger.info(f"Saved combined data to {filepath}")
    
    def get_latest_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get latest N days of data for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch
        
        Returns:
            DataFrame with recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_stock_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    def update_all_data(self):
        """Update data for all configured stocks"""
        logger.info("Starting full data update...")
        
        # Fetch NIFTY 50 data
        self.fetch_nifty50_data()
        
        # Fetch indices
        indices = self.symbols_config.get('indices', [])
        for index in indices:
            self.fetch_index_data(index)
        
        logger.info("Data update complete!")


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Fetch single stock data
    reliance_data = pipeline.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
    print("Reliance Data:")
    print(reliance_data.head())
    
    # Fetch NIFTY 50 index
    nifty_data = pipeline.fetch_index_data('^NSEI', start_date='2023-01-01')
    print("\nNIFTY 50 Index Data:")
    print(nifty_data.head())

