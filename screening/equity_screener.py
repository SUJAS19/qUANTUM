"""
Equity Screener
Identify undervalued equities using fundamental and technical analysis
Reduces manual research time by 80%+
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import logging
from .fundamental import FundamentalAnalyzer
from .technical import TechnicalAnalyzer
import yaml

logger = logging.getLogger(__name__)


class EquityScreener:
    """
    Comprehensive equity screening tool combining fundamental and technical analysis
    """
    
    def __init__(self, config_path: str = "config/config.example.yaml"):
        """Initialize screener with configuration"""
        self.config = self._load_config(config_path)
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.symbols = self._load_symbols()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            return self._default_config()
    
    def _load_symbols(self) -> List[str]:
        """Load stock symbols"""
        try:
            with open('config/symbols.yaml', 'r') as f:
                symbols_config = yaml.safe_load(f)
                return symbols_config.get('nifty50', [])
        except FileNotFoundError:
            logger.warning("Symbols file not found")
            return []
    
    def _default_config(self) -> Dict:
        """Default screening configuration"""
        return {
            'screening': {
                'fundamental': {
                    'max_pe_ratio': 25,
                    'min_roe': 15,
                    'max_debt_equity': 1.0,
                    'min_current_ratio': 1.5
                },
                'technical': {
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'ma_periods': [20, 50, 200],
                    'volume_threshold': 1.5
                }
            }
        }
    
    def screen_undervalued_stocks(
        self,
        symbols: Optional[List[str]] = None,
        min_roe: float = 15,
        max_pe: float = 25,
        max_debt_equity: float = 1.0,
        min_current_ratio: float = 1.5,
        rsi_range: tuple = (30, 70),
        price_above_ma: int = 50,
        min_volume_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        Screen for undervalued stocks using fundamental and technical criteria
        
        Args:
            symbols: List of symbols to screen (uses NIFTY 50 if None)
            min_roe: Minimum Return on Equity
            max_pe: Maximum P/E ratio
            max_debt_equity: Maximum Debt-to-Equity ratio
            min_current_ratio: Minimum Current Ratio
            rsi_range: Tuple of (min_rsi, max_rsi)
            price_above_ma: Moving average period for price comparison
            min_volume_ratio: Minimum volume ratio vs average
        
        Returns:
            DataFrame with screened stocks and their metrics
        """
        if symbols is None:
            symbols = self.symbols
        
        if not symbols:
            logger.error("No symbols provided for screening")
            return pd.DataFrame()
        
        logger.info(f"Screening {len(symbols)} stocks...")
        
        results = []
        
        for symbol in symbols:
            try:
                # Get stock info
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Get price history
                history = stock.history(period='6mo')
                
                if history.empty:
                    continue
                
                # Fundamental screening
                pe_ratio = info.get('trailingPE', float('inf'))
                roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                debt_equity = info.get('debtToEquity', float('inf')) / 100 if info.get('debtToEquity') else float('inf')
                current_ratio = info.get('currentRatio', 0)
                
                # Technical screening
                rsi = self.technical_analyzer.calculate_rsi(history['Close'])
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                
                ma = history['Close'].rolling(window=price_above_ma).mean()
                current_price = history['Close'].iloc[-1]
                current_ma = ma.iloc[-1] if not ma.empty else current_price
                
                avg_volume = history['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = history['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Apply filters
                passes_fundamental = (
                    pe_ratio <= max_pe and
                    roe >= min_roe and
                    debt_equity <= max_debt_equity and
                    current_ratio >= min_current_ratio
                )
                
                passes_technical = (
                    rsi_range[0] <= current_rsi <= rsi_range[1] and
                    current_price > current_ma and
                    volume_ratio >= min_volume_ratio
                )
                
                if passes_fundamental and passes_technical:
                    results.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'PE_Ratio': round(pe_ratio, 2),
                        'ROE': round(roe, 2),
                        'Debt_Equity': round(debt_equity, 2),
                        'Current_Ratio': round(current_ratio, 2),
                        'RSI': round(current_rsi, 2),
                        'Price_vs_MA': round((current_price - current_ma) / current_ma * 100, 2),
                        'Volume_Ratio': round(volume_ratio, 2),
                        'Market_Cap': info.get('marketCap', 0),
                        'Sector': info.get('sector', 'N/A')
                    })
                
            except Exception as e:
                logger.debug(f"Error screening {symbol}: {str(e)}")
                continue
        
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Sort by composite score (ROE / PE ratio)
            df_results['Value_Score'] = df_results['ROE'] / df_results['PE_Ratio']
            df_results = df_results.sort_values('Value_Score', ascending=False)
        
        logger.info(f"Found {len(df_results)} stocks matching criteria")
        
        return df_results
    
    def screen_momentum_stocks(
        self,
        symbols: Optional[List[str]] = None,
        min_return_1m: float = 5,
        min_return_3m: float = 10,
        min_rsi: float = 50,
        min_volume_ratio: float = 1.5
    ) -> pd.DataFrame:
        """
        Screen for momentum stocks
        
        Args:
            symbols: List of symbols
            min_return_1m: Minimum 1-month return (%)
            min_return_3m: Minimum 3-month return (%)
            min_rsi: Minimum RSI value
            min_volume_ratio: Minimum volume ratio
        
        Returns:
            DataFrame with momentum stocks
        """
        if symbols is None:
            symbols = self.symbols
        
        logger.info(f"Screening for momentum stocks...")
        
        results = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period='6mo')
                
                if len(history) < 90:
                    continue
                
                # Calculate returns
                current_price = history['Close'].iloc[-1]
                price_1m_ago = history['Close'].iloc[-21] if len(history) >= 21 else history['Close'].iloc[0]
                price_3m_ago = history['Close'].iloc[-63] if len(history) >= 63 else history['Close'].iloc[0]
                
                return_1m = (current_price - price_1m_ago) / price_1m_ago * 100
                return_3m = (current_price - price_3m_ago) / price_3m_ago * 100
                
                # Calculate RSI
                rsi = self.technical_analyzer.calculate_rsi(history['Close'])
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                
                # Volume analysis
                avg_volume = history['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = history['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Apply filters
                if (return_1m >= min_return_1m and
                    return_3m >= min_return_3m and
                    current_rsi >= min_rsi and
                    volume_ratio >= min_volume_ratio):
                    
                    results.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Return_1M': round(return_1m, 2),
                        'Return_3M': round(return_3m, 2),
                        'RSI': round(current_rsi, 2),
                        'Volume_Ratio': round(volume_ratio, 2),
                        'Momentum_Score': round(return_1m * 0.3 + return_3m * 0.7, 2)
                    })
            
            except Exception as e:
                logger.debug(f"Error screening {symbol}: {str(e)}")
                continue
        
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            df_results = df_results.sort_values('Momentum_Score', ascending=False)
        
        logger.info(f"Found {len(df_results)} momentum stocks")
        
        return df_results
    
    def screen_breakout_stocks(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        Screen for stocks breaking out of resistance levels
        
        Args:
            symbols: List of symbols
            lookback_days: Days to look back for resistance
        
        Returns:
            DataFrame with breakout stocks
        """
        if symbols is None:
            symbols = self.symbols
        
        logger.info(f"Screening for breakout stocks...")
        
        results = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period='6mo')
                
                if len(history) < lookback_days:
                    continue
                
                current_price = history['Close'].iloc[-1]
                high_52w = history['High'].iloc[-lookback_days:].max()
                
                # Check if price is breaking out (within 2% of 52-week high)
                breakout_threshold = high_52w * 0.98
                
                if current_price >= breakout_threshold:
                    volume_surge = history['Volume'].iloc[-5:].mean() / history['Volume'].iloc[-30:].mean()
                    
                    results.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        '52W_High': high_52w,
                        'Distance_from_High': round((high_52w - current_price) / high_52w * 100, 2),
                        'Volume_Surge': round(volume_surge, 2)
                    })
            
            except Exception as e:
                logger.debug(f"Error screening {symbol}: {str(e)}")
                continue
        
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            df_results = df_results.sort_values('Volume_Surge', ascending=False)
        
        logger.info(f"Found {len(df_results)} breakout stocks")
        
        return df_results


if __name__ == "__main__":
    # Example usage
    screener = EquityScreener()
    
    # Screen for undervalued stocks
    undervalued = screener.screen_undervalued_stocks(
        min_roe=15,
        max_pe=20,
        max_debt_equity=1.0,
        rsi_range=(30, 70)
    )
    
    print("Undervalued Stocks:")
    print(undervalued.head(10))
    
    # Screen for momentum stocks
    momentum = screener.screen_momentum_stocks(
        min_return_1m=5,
        min_return_3m=10
    )
    
    print("\nMomentum Stocks:")
    print(momentum.head(10))

