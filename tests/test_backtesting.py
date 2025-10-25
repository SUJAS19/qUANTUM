"""
Unit tests for backtesting engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from backtesting.backtest_engine import BacktestEngine, Trade
from backtesting.performance import PerformanceMetrics
from backtesting.risk_metrics import RiskMetrics


def test_backtest_engine_init():
    """Test backtest engine initialization"""
    engine = BacktestEngine(initial_capital=100000)
    assert engine.initial_capital == 100000
    assert engine.capital == 100000
    assert engine.commission == 0.001
    assert engine.slippage == 0.0005


def test_trade_creation():
    """Test trade object creation"""
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
    assert trade.exit_date is None
    assert trade.pnl is None


def test_trade_close_long():
    """Test closing a long trade"""
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
    assert trade.return_pct == 0.1  # 10% return


def test_trade_close_short():
    """Test closing a short trade"""
    trade = Trade(
        entry_date=datetime(2024, 1, 1),
        exit_date=None,
        entry_price=100.0,
        exit_price=None,
        quantity=10,
        direction='short'
    )
    
    trade.close_trade(datetime(2024, 1, 10), 95.0)
    
    assert trade.exit_price == 95.0
    assert trade.pnl == 50.0  # (100 - 95) * 10
    assert trade.return_pct == 0.05  # 5% return


def test_performance_metrics():
    """Test performance metrics calculation"""
    # Create mock backtest results
    equity_curve = [100000, 102000, 104000, 103000, 105000]
    
    class MockResults:
        def __init__(self):
            self.initial_capital = 100000
            self.final_capital = 105000
            self.equity_curve = equity_curve
            self.trades = []
    
    results = MockResults()
    metrics = PerformanceMetrics(results)
    
    total_return = metrics.total_return()
    assert total_return == 5.0  # 5% return


def test_risk_metrics():
    """Test risk metrics calculation"""
    equity_curve = [100000, 102000, 101000, 103000, 105000]
    
    class MockResults:
        def __init__(self):
            self.equity_curve = equity_curve
            self.trades = []
    
    results = MockResults()
    risk = RiskMetrics(results)
    
    vol = risk.volatility(annualized=False)
    assert vol >= 0  # Volatility should be non-negative

