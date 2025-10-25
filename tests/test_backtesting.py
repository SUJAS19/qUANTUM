"""
Unit tests for backtesting engine
"""

import pytest
import pandas as pd
import numpy as np
from backtesting.backtest_engine import BacktestEngine, Trade


def test_backtest_engine_init():
    """Test backtest engine initialization"""
    engine = BacktestEngine(initial_capital=100000)
    assert engine.initial_capital == 100000
    assert engine.capital == 100000


def test_trade_creation():
    """Test trade object creation"""
    from datetime import datetime
    
    trade = Trade(
        entry_date=datetime(2024, 1, 1),
        exit_date=None,
        entry_price=100.0,
        exit_price=None,
        quantity=10,
        direction='long'
    )
    
    assert trade.entry_price == 100.0
    assert trade.quantity == 10
    assert trade.direction == 'long'


def test_trade_close():
    """Test closing a trade"""
    from datetime import datetime
    
    trade = Trade(
        entry_date=datetime(2024, 1, 1),
        exit_date=None,
        entry_price=100.0,
        exit_price=None,
        quantity=10,
        direction='long'
    )
    
    trade.close_trade(datetime(2024, 1, 10), 110.0)
    
    assert trade.exit_price == 110.0
    assert trade.pnl == 100.0  # (110 - 100) * 10

