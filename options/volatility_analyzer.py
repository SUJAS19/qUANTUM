"""
Volatility Analyzer
Model volatility patterns and analyze implied vs historical volatility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import logging
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Analyze and model volatility patterns for risk management and trading decisions
    """
    
    @staticmethod
    def calculate_historical_volatility(
        prices: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate historical volatility (realized volatility)
        
        Args:
            prices: Price series
            window: Rolling window in days
            annualize: Whether to annualize volatility
        
        Returns:
            Historical volatility series
        """
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Parkinson's volatility (uses high-low range)
        More efficient than close-to-close volatility
        
        Args:
            high: High prices
            low: Low prices
            window: Rolling window
            annualize: Whether to annualize
        
        Returns:
            Parkinson volatility series
        """
        hl_ratio = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
        volatility = np.sqrt(parkinson_var.rolling(window=window).mean())
        
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_garman_klass_volatility(
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator
        Uses OHLC data for more accurate estimation
        
        Args:
            open_price: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window
            annualize: Whether to annualize
        
        Returns:
            Garman-Klass volatility series
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        volatility = np.sqrt(gk_var.rolling(window=window).mean())
        
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_ewma_volatility(
        prices: pd.Series,
        span: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility
        
        Args:
            prices: Price series
            span: Span for EWMA
            annualize: Whether to annualize
        
        Returns:
            EWMA volatility series
        """
        returns = np.log(prices / prices.shift(1))
        volatility = returns.ewm(span=span).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_garch_volatility(
        returns: pd.Series,
        p: int = 1,
        q: int = 1
    ) -> Optional[pd.Series]:
        """
        Calculate GARCH(p,q) volatility forecast
        
        Note: Requires arch library
        
        Args:
            returns: Return series
            p: GARCH lag order
            q: ARCH lag order
        
        Returns:
            GARCH volatility forecast or None if library not available
        """
        try:
            from arch import arch_model
            
            # Fit GARCH model
            model = arch_model(returns.dropna() * 100, vol='Garch', p=p, q=q)
            results = model.fit(disp='off')
            
            # Get conditional volatility
            volatility = results.conditional_volatility / 100
            
            return volatility
        
        except ImportError:
            logger.warning("arch library not installed, GARCH calculation skipped")
            return None
    
    @staticmethod
    def calculate_volatility_cone(
        prices: pd.Series,
        windows: list = [10, 20, 30, 60, 90, 120]
    ) -> pd.DataFrame:
        """
        Calculate volatility cone (percentiles across different windows)
        
        Args:
            prices: Price series
            windows: List of window sizes
        
        Returns:
            DataFrame with volatility cone data
        """
        percentiles = [10, 25, 50, 75, 90]
        cone_data = []
        
        for window in windows:
            vol = VolatilityAnalyzer.calculate_historical_volatility(prices, window)
            
            if not vol.empty:
                stats = {
                    'window': window,
                    'current': vol.iloc[-1] if not pd.isna(vol.iloc[-1]) else 0
                }
                
                for p in percentiles:
                    stats[f'p{p}'] = np.percentile(vol.dropna(), p)
                
                cone_data.append(stats)
        
        return pd.DataFrame(cone_data)
    
    @staticmethod
    def calculate_volatility_surface(
        option_chain: pd.DataFrame,
        spot_price: float
    ) -> pd.DataFrame:
        """
        Calculate implied volatility surface
        
        Args:
            option_chain: DataFrame with columns ['Strike', 'Expiry', 'IV', 'Type']
            spot_price: Current spot price
        
        Returns:
            DataFrame with volatility surface
        """
        if option_chain.empty:
            logger.warning("Empty option chain provided")
            return pd.DataFrame()
        
        # Calculate moneyness
        option_chain['Moneyness'] = option_chain['Strike'] / spot_price
        
        # Pivot to create surface
        surface = option_chain.pivot_table(
            values='IV',
            index='Moneyness',
            columns='Expiry',
            aggfunc='mean'
        )
        
        return surface
    
    @staticmethod
    def calculate_volatility_smile(
        strikes: list,
        ivs: list,
        spot_price: float
    ) -> pd.DataFrame:
        """
        Calculate volatility smile
        
        Args:
            strikes: List of strike prices
            ivs: List of implied volatilities
            spot_price: Current spot price
        
        Returns:
            DataFrame with smile data
        """
        moneyness = [strike / spot_price for strike in strikes]
        
        smile_df = pd.DataFrame({
            'Strike': strikes,
            'Moneyness': moneyness,
            'IV': ivs
        })
        
        smile_df = smile_df.sort_values('Moneyness')
        
        return smile_df
    
    @staticmethod
    def calculate_volatility_skew(smile_df: pd.DataFrame) -> float:
        """
        Calculate volatility skew (25-delta put IV - 25-delta call IV)
        
        Args:
            smile_df: DataFrame with volatility smile data
        
        Returns:
            Skew value
        """
        if len(smile_df) < 2:
            return 0.0
        
        # Approximate: use OTM put and call IVs
        otm_put_iv = smile_df[smile_df['Moneyness'] < 0.95]['IV'].mean()
        otm_call_iv = smile_df[smile_df['Moneyness'] > 1.05]['IV'].mean()
        
        skew = otm_put_iv - otm_call_iv
        
        return skew if not pd.isna(skew) else 0.0
    
    @staticmethod
    def plot_volatility_cone(cone_df: pd.DataFrame):
        """Plot volatility cone"""
        plt.figure(figsize=(12, 6))
        
        plt.fill_between(cone_df['window'], cone_df['p10'], cone_df['p90'], 
                         alpha=0.2, label='10th-90th percentile')
        plt.fill_between(cone_df['window'], cone_df['p25'], cone_df['p75'], 
                         alpha=0.3, label='25th-75th percentile')
        plt.plot(cone_df['window'], cone_df['p50'], 'b-', label='Median', linewidth=2)
        plt.plot(cone_df['window'], cone_df['current'], 'ro-', label='Current', linewidth=2)
        
        plt.xlabel('Window (days)')
        plt.ylabel('Annualized Volatility')
        plt.title('Volatility Cone')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_volatility_smile(smile_df: pd.DataFrame, title: str = "Volatility Smile"):
        """Plot volatility smile"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(smile_df['Moneyness'], smile_df['IV'] * 100, 'b-o', linewidth=2)
        plt.axvline(x=1.0, color='r', linestyle='--', label='ATM', alpha=0.7)
        
        plt.xlabel('Moneyness (Strike / Spot)')
        plt.ylabel('Implied Volatility (%)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_volatility_surface(surface_df: pd.DataFrame):
        """Plot 3D volatility surface"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh
        X = surface_df.columns.values  # Expiry
        Y = surface_df.index.values    # Moneyness
        X, Y = np.meshgrid(X, Y)
        Z = surface_df.values * 100    # IV in percentage
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Moneyness')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title('Volatility Surface')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    
    @staticmethod
    def analyze_volatility_regime(
        historical_vol: pd.Series,
        current_vol: float
    ) -> Dict[str, str]:
        """
        Determine current volatility regime
        
        Args:
            historical_vol: Historical volatility series
            current_vol: Current volatility level
        
        Returns:
            Dictionary with regime analysis
        """
        percentile = (historical_vol < current_vol).sum() / len(historical_vol) * 100
        
        if percentile < 20:
            regime = "Low Volatility"
            interpretation = "Current volatility is in the bottom 20%. Consider long volatility strategies."
        elif percentile < 40:
            regime = "Below Average Volatility"
            interpretation = "Volatility is below average. Moderate opportunity for volatility plays."
        elif percentile < 60:
            regime = "Average Volatility"
            interpretation = "Volatility is near historical average. Neutral environment."
        elif percentile < 80:
            regime = "Above Average Volatility"
            interpretation = "Elevated volatility. Consider defensive strategies."
        else:
            regime = "High Volatility"
            interpretation = "Volatility in top 20%. Consider short volatility or protective strategies."
        
        return {
            'regime': regime,
            'percentile': round(percentile, 2),
            'interpretation': interpretation,
            'current_vol': round(current_vol * 100, 2),
            'historical_mean': round(historical_vol.mean() * 100, 2),
            'historical_std': round(historical_vol.std() * 100, 2)
        }


if __name__ == "__main__":
    # Example usage
    from data.data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01')
    
    analyzer = VolatilityAnalyzer()
    
    # Calculate different volatility measures
    hist_vol = analyzer.calculate_historical_volatility(data['Close'])
    park_vol = analyzer.calculate_parkinson_volatility(data['High'], data['Low'])
    
    print("Historical Volatility (30-day):", f"{hist_vol.iloc[-1]*100:.2f}%")
    print("Parkinson Volatility (30-day):", f"{park_vol.iloc[-1]*100:.2f}%")
    
    # Calculate volatility cone
    cone = analyzer.calculate_volatility_cone(data['Close'])
    print("\nVolatility Cone:")
    print(cone)
    
    # Analyze volatility regime
    regime = analyzer.analyze_volatility_regime(hist_vol.dropna(), hist_vol.iloc[-1])
    print(f"\nVolatility Regime: {regime['regime']}")
    print(f"Percentile: {regime['percentile']}%")
    print(f"Interpretation: {regime['interpretation']}")

