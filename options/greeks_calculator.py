"""
Options Greeks Calculator
Calculate Delta, Gamma, Theta, Vega, and Rho for options
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes"""
        return GreeksCalculator._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def black_scholes_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate option price using Black-Scholes model
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        if T <= 0:
            # Handle expired options
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = GreeksCalculator._d1(S, K, T, r, sigma)
        d2 = GreeksCalculator._d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Delta (rate of change of option price with respect to underlying price)
        
        Delta ranges:
        - Call: 0 to 1
        - Put: -1 to 0
        
        Returns:
            Delta value
        """
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = GreeksCalculator._d1(S, K, T, r, sigma)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = -norm.cdf(-d1)
        
        return delta
    
    @staticmethod
    def calculate_gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Gamma (rate of change of delta with respect to underlying price)
        
        Same for both calls and puts
        
        Returns:
            Gamma value
        """
        if T <= 0:
            return 0.0
        
        d1 = GreeksCalculator._d1(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma
    
    @staticmethod
    def calculate_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Theta (rate of change of option price with respect to time)
        
        Typically negative (options lose value as time passes)
        Returned as value per day
        
        Returns:
            Theta value (per day)
        """
        if T <= 0:
            return 0.0
        
        d1 = GreeksCalculator._d1(S, K, T, r, sigma)
        d2 = GreeksCalculator._d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Convert to per-day theta
        theta_per_day = theta / 365
        
        return theta_per_day
    
    @staticmethod
    def calculate_vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Vega (rate of change of option price with respect to volatility)
        
        Same for both calls and puts
        Returned as change per 1% change in volatility
        
        Returns:
            Vega value
        """
        if T <= 0:
            return 0.0
        
        d1 = GreeksCalculator._d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Convert to per 1% change
        vega = vega / 100
        
        return vega
    
    @staticmethod
    def calculate_rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Rho (rate of change of option price with respect to interest rate)
        
        Returned as change per 1% change in interest rate
        
        Returns:
            Rho value
        """
        if T <= 0:
            return 0.0
        
        d2 = GreeksCalculator._d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to per 1% change
        rho = rho / 100
        
        return rho
    
    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annual)
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with all Greeks
        """
        price = GreeksCalculator.black_scholes_price(S, K, T, r, sigma, option_type)
        delta = GreeksCalculator.calculate_delta(S, K, T, r, sigma, option_type)
        gamma = GreeksCalculator.calculate_gamma(S, K, T, r, sigma)
        theta = GreeksCalculator.calculate_theta(S, K, T, r, sigma, option_type)
        vega = GreeksCalculator.calculate_vega(S, K, T, r, sigma)
        rho = GreeksCalculator.calculate_rho(S, K, T, r, sigma, option_type)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'spot_price': S,
            'strike_price': K,
            'time_to_expiry': T,
            'volatility': sigma,
            'option_type': option_type
        }
    
    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            option_price: Market price of option
            S: Spot price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        
        Returns:
            Implied volatility or None if calculation fails
        """
        if T <= 0:
            return None
        
        # Initial guess
        sigma = 0.3
        
        for i in range(max_iterations):
            # Calculate option price with current sigma
            price = GreeksCalculator.black_scholes_price(S, K, T, r, sigma, option_type)
            
            # Calculate vega
            vega = GreeksCalculator.calculate_vega(S, K, T, r, sigma) * 100  # Undo the /100
            
            # Calculate difference
            diff = option_price - price
            
            # Check convergence
            if abs(diff) < tolerance:
                return sigma
            
            # Avoid division by zero
            if vega == 0:
                return None
            
            # Newton-Raphson update
            sigma = sigma + diff / vega
            
            # Keep sigma positive
            if sigma <= 0:
                sigma = 0.01
        
        logger.warning("Implied volatility calculation did not converge")
        return sigma


if __name__ == "__main__":
    # Example usage
    calc = GreeksCalculator()
    
    # Example: NIFTY 50 call option
    S = 19500  # Spot price
    K = 19500  # Strike price (ATM)
    T = 30 / 365  # 30 days to expiration
    r = 0.07  # 7% risk-free rate
    sigma = 0.15  # 15% volatility
    
    # Calculate all Greeks
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    
    print("Option Greeks:")
    print(f"Price: ₹{greeks['price']:.2f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Theta: ₹{greeks['theta']:.4f} per day")
    print(f"Vega: ₹{greeks['vega']:.4f} per 1% IV change")
    print(f"Rho: ₹{greeks['rho']:.4f} per 1% rate change")
    
    # Calculate implied volatility
    market_price = 250
    iv = calc.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"\nImplied Volatility: {iv*100:.2f}%")

