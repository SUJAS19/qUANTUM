# Quick Start Guide - qUANTUM

Welcome to qUANTUM! This guide will help you get started with the quantitative trading framework.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/SUJAS19/qUANTUM.git
cd qUANTUM
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up configuration**
```bash
cp config/config.example.yaml config/config.yaml
```

## 5-Minute Tutorial

### 1. Fetch Market Data

```python
from data.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Fetch stock data
data = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01')
print(data.tail())
```

### 2. Screen for Undervalued Stocks

```python
from screening.equity_screener import EquityScreener

# Initialize screener
screener = EquityScreener()

# Find undervalued stocks
results = screener.screen_undervalued_stocks(
    min_roe=15,
    max_pe=20,
    rsi_range=(30, 70)
)

print(results.head())
```

### 3. Backtest a Strategy

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy

# Create strategy
strategy = MeanReversionStrategy(window=20)

# Run backtest
engine = BacktestEngine(initial_capital=100000)
results = engine.run(strategy, data, symbol='TCS.NS')

# View results
results.print_summary()
results.plot_performance()
```

### 4. Forecast Prices with ML

```python
from ml.price_forecaster import PriceForecaster

# Train model
forecaster = PriceForecaster(model_type='random_forest')
metrics = forecaster.train(data)

print(f"R²: {metrics['r2']:.4f}")
print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
```

### 5. Analyze Options

```python
from options.greeks_calculator import GreeksCalculator

calc = GreeksCalculator()

# Calculate Greeks
greeks = calc.calculate_all_greeks(
    S=19500,    # Spot price
    K=19500,    # Strike
    T=30/365,   # 30 days to expiry
    r=0.07,     # 7% risk-free rate
    sigma=0.15, # 15% volatility
    option_type='call'
)

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Theta: {greeks['theta']:.4f}")
```

### 6. Volatility Analysis

```python
from options.volatility_analyzer import VolatilityAnalyzer

analyzer = VolatilityAnalyzer()

# Calculate historical volatility
hist_vol = analyzer.calculate_historical_volatility(data['Close'])

# Analyze volatility regime
regime = analyzer.analyze_volatility_regime(
    hist_vol.dropna(), 
    hist_vol.iloc[-1]
)

print(f"Regime: {regime['regime']}")
print(f"Interpretation: {regime['interpretation']}")
```

## Running Examples

Run the comprehensive example script:

```bash
cd examples
python basic_usage.py
```

## Next Steps

- **Explore Strategies**: Check out `strategies/` for different trading strategies
- **Customize Configuration**: Edit `config/config.yaml` to match your preferences
- **Build Your Own Strategy**: Extend the `Strategy` base class
- **Add New Features**: Use `ml.features.FeatureEngineer` to create custom indicators

## Common Use Cases

### Finding Trading Opportunities

```python
# Screen for momentum stocks
screener = EquityScreener()
momentum_stocks = screener.screen_momentum_stocks(
    min_return_1m=5,
    min_return_3m=10
)
```

### Optimizing Strategy Parameters

```python
# Test different parameters
for window in [10, 20, 30, 50]:
    strategy = MeanReversionStrategy(window=window)
    results = engine.run(strategy, data)
    print(f"Window {window}: Sharpe = {results.performance.sharpe_ratio():.2f}")
```

### Building a Trading System

```python
# 1. Screen stocks
screener = EquityScreener()
candidates = screener.screen_undervalued_stocks()

# 2. Backtest each
for symbol in candidates['Symbol']:
    data = pipeline.fetch_stock_data(symbol)
    results = engine.run(strategy, data, symbol=symbol)
    
    if results.performance.sharpe_ratio() > 1.5:
        print(f"{symbol} passed screening!")
```

## Tips & Best Practices

1. **Start Small**: Test with small capital before going live
2. **Diversify**: Don't put all eggs in one basket
3. **Risk Management**: Always use stop losses and position sizing
4. **Backtest Thoroughly**: Test across different market conditions
5. **Stay Updated**: Markets change, update your data regularly

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Issues**: Report bugs on GitHub
- **Examples**: See `examples/` for more use cases

## Disclaimer

⚠️ **For Educational Purposes Only**

This framework is for educational and research purposes. Trading involves substantial risk. Always:
- Do your own research
- Consult with financial advisors
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

---

Happy Trading! 📈⚡

