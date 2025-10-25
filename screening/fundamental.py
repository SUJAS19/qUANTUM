"""
Fundamental Analysis Module
Analyze fundamental metrics for stock valuation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """
    Fundamental analysis tools for stock evaluation
    """
    
    @staticmethod
    def get_fundamental_data(symbol: str) -> Dict:
        """
        Fetch fundamental data for a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                # Valuation metrics
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                
                # Profitability metrics
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'roic': info.get('returnOnCapital'),
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # Financial health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'free_cash_flow': info.get('freeCashflow'),
                
                # Dividend metrics
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                
                # Market data
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'beta': info.get('beta'),
                
                # Company info
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
        
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_intrinsic_value_dcf(
        free_cash_flow: float,
        growth_rate: float,
        discount_rate: float,
        terminal_growth_rate: float = 0.03,
        projection_years: int = 5,
        shares_outstanding: float = 1
    ) -> float:
        """
        Calculate intrinsic value using Discounted Cash Flow (DCF) model
        
        Args:
            free_cash_flow: Current free cash flow
            growth_rate: Expected growth rate for projection period
            discount_rate: Discount rate (WACC)
            terminal_growth_rate: Perpetual growth rate
            projection_years: Number of years to project
            shares_outstanding: Number of shares outstanding
        
        Returns:
            Intrinsic value per share
        """
        if shares_outstanding == 0:
            return 0.0
        
        # Project future cash flows
        projected_fcf = []
        for year in range(1, projection_years + 1):
            fcf = free_cash_flow * ((1 + growth_rate) ** year)
            discounted_fcf = fcf / ((1 + discount_rate) ** year)
            projected_fcf.append(discounted_fcf)
        
        # Calculate terminal value
        terminal_fcf = free_cash_flow * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # Calculate enterprise value
        enterprise_value = sum(projected_fcf) + discounted_terminal_value
        
        # Calculate intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding
        
        return intrinsic_value
    
    @staticmethod
    def calculate_graham_number(eps: float, book_value_per_share: float) -> float:
        """
        Calculate Graham Number for value investing
        
        Args:
            eps: Earnings per share
            book_value_per_share: Book value per share
        
        Returns:
            Graham number
        """
        if eps <= 0 or book_value_per_share <= 0:
            return 0.0
        
        return np.sqrt(22.5 * eps * book_value_per_share)
    
    @staticmethod
    def calculate_altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_value_equity: float,
        total_liabilities: float,
        total_assets: float,
        sales: float
    ) -> float:
        """
        Calculate Altman Z-Score for bankruptcy prediction
        
        Z > 2.99: Safe zone
        1.81 < Z < 2.99: Grey zone
        Z < 1.81: Distress zone
        
        Returns:
            Altman Z-Score
        """
        if total_assets == 0:
            return 0.0
        
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_value_equity / total_liabilities if total_liabilities > 0 else 0
        x5 = sales / total_assets
        
        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        
        return z_score
    
    @staticmethod
    def calculate_piotroski_f_score(fundamentals: Dict) -> int:
        """
        Calculate Piotroski F-Score (0-9, higher is better)
        
        Args:
            fundamentals: Dictionary with fundamental metrics
        
        Returns:
            F-Score (0-9)
        """
        score = 0
        
        # Profitability (4 points max)
        if fundamentals.get('roe', 0) > 0:
            score += 1
        if fundamentals.get('roa', 0) > 0:
            score += 1
        if fundamentals.get('free_cash_flow', 0) > 0:
            score += 1
        if fundamentals.get('operating_margins', 0) > 0:
            score += 1
        
        # Leverage/Liquidity (3 points max)
        if fundamentals.get('current_ratio', 0) > 1.5:
            score += 1
        if fundamentals.get('quick_ratio', 0) > 1.0:
            score += 1
        if fundamentals.get('debt_to_equity', float('inf')) < 0.5:
            score += 1
        
        # Operating Efficiency (2 points max)
        if fundamentals.get('profit_margins', 0) > 0.1:
            score += 1
        if fundamentals.get('operating_margins', 0) > 0.15:
            score += 1
        
        return score
    
    @staticmethod
    def analyze_financial_health(symbol: str) -> Dict:
        """
        Comprehensive financial health analysis
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with financial health metrics and ratings
        """
        fundamentals = FundamentalAnalyzer.get_fundamental_data(symbol)
        
        if not fundamentals:
            return {}
        
        # Calculate scores
        f_score = FundamentalAnalyzer.calculate_piotroski_f_score(fundamentals)
        
        # Determine ratings
        valuation_rating = "Undervalued" if fundamentals.get('pe_ratio', float('inf')) < 20 else "Overvalued"
        profitability_rating = "Strong" if fundamentals.get('roe', 0) > 0.15 else "Weak"
        growth_rating = "High" if fundamentals.get('revenue_growth', 0) > 0.15 else "Low"
        
        # Debt analysis
        debt_equity = fundamentals.get('debt_to_equity', 0)
        if debt_equity:
            debt_equity = debt_equity / 100  # Convert from percentage
        debt_rating = "Low" if debt_equity < 0.5 else ("Medium" if debt_equity < 1.0 else "High")
        
        return {
            'symbol': symbol,
            'piotroski_f_score': f_score,
            'valuation_rating': valuation_rating,
            'profitability_rating': profitability_rating,
            'growth_rating': growth_rating,
            'debt_rating': debt_rating,
            'fundamentals': fundamentals
        }


if __name__ == "__main__":
    # Example usage
    analyzer = FundamentalAnalyzer()
    
    # Analyze a stock
    analysis = analyzer.analyze_financial_health('TCS.NS')
    
    print("Financial Health Analysis for TCS:")
    print(f"Piotroski F-Score: {analysis.get('piotroski_f_score')}")
    print(f"Valuation: {analysis.get('valuation_rating')}")
    print(f"Profitability: {analysis.get('profitability_rating')}")
    print(f"Growth: {analysis.get('growth_rating')}")
    print(f"Debt: {analysis.get('debt_rating')}")

