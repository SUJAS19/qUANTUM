"""
Option Chain Analyzer
Analyze option chain data for NIFTY 50 and other indices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptionChainAnalyzer:
    """
    Analyze option chain for trading opportunities and market sentiment
    """
    
    @staticmethod
    def calculate_pcr(call_oi: float, put_oi: float) -> float:
        """
        Calculate Put-Call Ratio (PCR)
        
        PCR > 1: Bullish (more puts than calls)
        PCR < 1: Bearish (more calls than puts)
        
        Args:
            call_oi: Call open interest
            put_oi: Put open interest
        
        Returns:
            PCR value
        """
        if call_oi == 0:
            return 0.0
        return put_oi / call_oi
    
    @staticmethod
    def find_max_pain(option_chain: pd.DataFrame, spot_price: float) -> float:
        """
        Calculate Max Pain strike price
        (Strike where option writers lose the least)
        
        Args:
            option_chain: DataFrame with columns ['Strike', 'Call_OI', 'Put_OI']
            spot_price: Current spot price
        
        Returns:
            Max pain strike
        """
        if option_chain.empty:
            return spot_price
        
        strikes = option_chain['Strike'].values
        call_oi = option_chain['Call_OI'].values
        put_oi = option_chain['Put_OI'].values
        
        total_pain = []
        
        for strike in strikes:
            # Calculate pain for calls
            call_pain = np.sum(np.maximum(strike - strikes, 0) * call_oi)
            
            # Calculate pain for puts
            put_pain = np.sum(np.maximum(strikes - strike, 0) * put_oi)
            
            # Total pain
            total_pain.append(call_pain + put_pain)
        
        # Find strike with minimum pain
        max_pain_idx = np.argmin(total_pain)
        max_pain_strike = strikes[max_pain_idx]
        
        return max_pain_strike
    
    @staticmethod
    def identify_support_resistance(
        option_chain: pd.DataFrame,
        num_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels based on OI
        
        Args:
            option_chain: DataFrame with option chain data
            num_levels: Number of levels to identify
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if option_chain.empty:
            return [], []
        
        # Support: Highest put OI
        put_oi_sorted = option_chain.nlargest(num_levels, 'Put_OI')
        support_levels = sorted(put_oi_sorted['Strike'].tolist())
        
        # Resistance: Highest call OI
        call_oi_sorted = option_chain.nlargest(num_levels, 'Call_OI')
        resistance_levels = sorted(call_oi_sorted['Strike'].tolist(), reverse=True)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def calculate_oi_change_signals(
        current_chain: pd.DataFrame,
        previous_chain: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate OI change signals
        
        Build-up interpretations:
        - Long Build-up: Price up + Call OI up
        - Short Build-up: Price down + Put OI up
        - Long Unwinding: Price down + Call OI down
        - Short Covering: Price up + Put OI down
        
        Args:
            current_chain: Current option chain
            previous_chain: Previous option chain
        
        Returns:
            DataFrame with OI change analysis
        """
        if current_chain.empty or previous_chain.empty:
            return pd.DataFrame()
        
        # Merge chains
        merged = current_chain.merge(
            previous_chain,
            on='Strike',
            suffixes=('_curr', '_prev')
        )
        
        # Calculate changes
        merged['Call_OI_Change'] = merged['Call_OI_curr'] - merged['Call_OI_prev']
        merged['Put_OI_Change'] = merged['Put_OI_curr'] - merged['Put_OI_prev']
        merged['Call_OI_Change_Pct'] = (merged['Call_OI_Change'] / merged['Call_OI_prev']) * 100
        merged['Put_OI_Change_Pct'] = (merged['Put_OI_Change'] / merged['Put_OI_prev']) * 100
        
        # Identify significant changes (>20%)
        merged['Significant_Call_Change'] = abs(merged['Call_OI_Change_Pct']) > 20
        merged['Significant_Put_Change'] = abs(merged['Put_OI_Change_Pct']) > 20
        
        return merged[['Strike', 'Call_OI_Change', 'Put_OI_Change', 
                       'Call_OI_Change_Pct', 'Put_OI_Change_Pct',
                       'Significant_Call_Change', 'Significant_Put_Change']]
    
    @staticmethod
    def identify_option_strategies(
        option_chain: pd.DataFrame,
        spot_price: float,
        max_loss_tolerance: float = 5000
    ) -> List[Dict]:
        """
        Identify potential option strategies based on current market conditions
        
        Args:
            option_chain: Option chain data
            spot_price: Current spot price
            max_loss_tolerance: Maximum loss tolerance
        
        Returns:
            List of strategy recommendations
        """
        strategies = []
        
        if option_chain.empty:
            return strategies
        
        # Find ATM strike
        atm_strike = option_chain.iloc[(option_chain['Strike'] - spot_price).abs().argsort()[:1]]['Strike'].values[0]
        
        # Calculate PCR
        total_call_oi = option_chain['Call_OI'].sum()
        total_put_oi = option_chain['Put_OI'].sum()
        pcr = OptionChainAnalyzer.calculate_pcr(total_call_oi, total_put_oi)
        
        # Strategy 1: Bull Call Spread (if moderately bullish)
        if 0.8 < pcr < 1.2:
            strategies.append({
                'strategy': 'Bull Call Spread',
                'sentiment': 'Moderately Bullish',
                'description': f'Buy {atm_strike} Call, Sell {atm_strike + 100} Call',
                'risk': 'Limited',
                'reward': 'Limited',
                'pcr': pcr
            })
        
        # Strategy 2: Bear Put Spread (if moderately bearish)
        if pcr < 0.8:
            strategies.append({
                'strategy': 'Bear Put Spread',
                'sentiment': 'Moderately Bearish',
                'description': f'Buy {atm_strike} Put, Sell {atm_strike - 100} Put',
                'risk': 'Limited',
                'reward': 'Limited',
                'pcr': pcr
            })
        
        # Strategy 3: Iron Condor (if range-bound)
        if 0.9 < pcr < 1.1:
            strategies.append({
                'strategy': 'Iron Condor',
                'sentiment': 'Range-bound/Neutral',
                'description': f'Sell {atm_strike - 100} Put, Buy {atm_strike - 200} Put, Sell {atm_strike + 100} Call, Buy {atm_strike + 200} Call',
                'risk': 'Limited',
                'reward': 'Limited',
                'pcr': pcr
            })
        
        # Strategy 4: Straddle (if expecting high volatility)
        strategies.append({
            'strategy': 'Long Straddle',
            'sentiment': 'High Volatility Expected',
            'description': f'Buy {atm_strike} Call and Put',
            'risk': 'Limited to premium paid',
            'reward': 'Unlimited',
            'pcr': pcr
        })
        
        return strategies
    
    @staticmethod
    def calculate_gamma_exposure(
        option_chain: pd.DataFrame,
        spot_price: float
    ) -> pd.DataFrame:
        """
        Calculate dealer gamma exposure across strikes
        
        Args:
            option_chain: Option chain with OI data
            spot_price: Current spot price
        
        Returns:
            DataFrame with gamma exposure by strike
        """
        if option_chain.empty:
            return pd.DataFrame()
        
        # Simplified gamma calculation (actual would use Greeks calculator)
        option_chain['Distance_from_Spot'] = abs(option_chain['Strike'] - spot_price)
        
        # Gamma peaks at ATM
        option_chain['Approx_Gamma'] = np.exp(-option_chain['Distance_from_Spot'] / (spot_price * 0.1))
        
        # Net gamma exposure (calls positive, puts negative for dealers)
        option_chain['Call_Gamma_Exposure'] = -option_chain['Call_OI'] * option_chain['Approx_Gamma']
        option_chain['Put_Gamma_Exposure'] = option_chain['Put_OI'] * option_chain['Approx_Gamma']
        option_chain['Net_Gamma_Exposure'] = (option_chain['Call_Gamma_Exposure'] + 
                                               option_chain['Put_Gamma_Exposure'])
        
        return option_chain[['Strike', 'Call_Gamma_Exposure', 'Put_Gamma_Exposure', 
                             'Net_Gamma_Exposure']].sort_values('Strike')
    
    @staticmethod
    def analyze_option_chain(option_chain: pd.DataFrame, spot_price: float) -> Dict:
        """
        Comprehensive option chain analysis
        
        Args:
            option_chain: Option chain data
            spot_price: Current spot price
        
        Returns:
            Dictionary with complete analysis
        """
        if option_chain.empty:
            return {}
        
        # Calculate metrics
        total_call_oi = option_chain['Call_OI'].sum()
        total_put_oi = option_chain['Put_OI'].sum()
        pcr = OptionChainAnalyzer.calculate_pcr(total_call_oi, total_put_oi)
        
        max_pain = OptionChainAnalyzer.find_max_pain(option_chain, spot_price)
        support, resistance = OptionChainAnalyzer.identify_support_resistance(option_chain)
        
        # Market sentiment
        if pcr > 1.2:
            sentiment = "Bullish"
        elif pcr < 0.8:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        return {
            'spot_price': spot_price,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'pcr': round(pcr, 3),
            'sentiment': sentiment,
            'max_pain': max_pain,
            'support_levels': support,
            'resistance_levels': resistance,
            'distance_from_max_pain': round(((spot_price - max_pain) / spot_price) * 100, 2)
        }


if __name__ == "__main__":
    # Example usage with sample data
    sample_chain = pd.DataFrame({
        'Strike': [19000, 19100, 19200, 19300, 19400, 19500, 19600, 19700, 19800],
        'Call_OI': [50000, 45000, 40000, 35000, 30000, 60000, 40000, 35000, 30000],
        'Put_OI': [30000, 35000, 40000, 45000, 50000, 55000, 35000, 30000, 25000],
        'Call_Volume': [5000, 4500, 4000, 3500, 3000, 6000, 4000, 3500, 3000],
        'Put_Volume': [3000, 3500, 4000, 4500, 5000, 5500, 3500, 3000, 2500]
    })
    
    spot_price = 19500
    
    analyzer = OptionChainAnalyzer()
    
    # Comprehensive analysis
    analysis = analyzer.analyze_option_chain(sample_chain, spot_price)
    
    print("Option Chain Analysis:")
    print(f"Spot Price: {analysis['spot_price']}")
    print(f"PCR: {analysis['pcr']}")
    print(f"Sentiment: {analysis['sentiment']}")
    print(f"Max Pain: {analysis['max_pain']}")
    print(f"Support Levels: {analysis['support_levels']}")
    print(f"Resistance Levels: {analysis['resistance_levels']}")
    
    # Strategy recommendations
    strategies = analyzer.identify_option_strategies(sample_chain, spot_price)
    print("\nStrategy Recommendations:")
    for strategy in strategies:
        print(f"- {strategy['strategy']} ({strategy['sentiment']})")

