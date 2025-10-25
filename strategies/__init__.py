"""
Trading Strategies module
Various quantitative trading strategies for Indian equities
"""

from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .pairs_trading import PairsTradingStrategy

__all__ = ['MeanReversionStrategy', 'MomentumStrategy', 'PairsTradingStrategy']

