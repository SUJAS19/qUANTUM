# qUANTUM - Quantitative Trading Framework 📈

A comprehensive quantitative trading framework for Indian equities and NIFTY 50 options. Features automated data pipelines, backtesting engine, screening tools, volatility modeling, and ML-based price forecasting.

## 🚀 Key Features

### 1. **Backtesting Framework**
- Multiple quantitative trading strategies for Indian equities and NIFTY 50 options
- Robust strategy evaluation with performance metrics (Sharpe ratio, drawdown, win rate, etc.)
- Support for both equity and derivatives strategies

### 2. **Automated Data Pipeline**
- Automated collection and cleaning of daily market data (OHLCV)
- Derivatives Greeks calculation (Delta, Gamma, Theta, Vega)
- Coverage for 500+ securities
- Data validation and quality checks

### 3. **Equity Screening Tool**
- Fundamental analysis (P/E, P/B, ROE, debt ratios, etc.)
- Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
- Automated identification of undervalued equities
- 80%+ reduction in manual research time

### 4. **Volatility & Options Analysis**
- Volatility pattern modeling (historical and implied volatility)
- Option chain analysis for NIFTY 50
- Greeks analysis for risk management
- Opportunity identification in derivatives market

### 5. **Machine Learning Models**
- Linear and polynomial regression for trend analysis
- Time-series forecasting (ARIMA, Prophet)
- LSTM networks for price prediction
- Feature engineering for technical indicators

## 📋 Prerequisites

- Python 3.8 or higher
- Active internet connection for data fetching
- Optional: API keys for data providers (NSE, Yahoo Finance, etc.)

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/SUJAS19/qUANTUM.git
cd qUANTUM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

## 📖 Quick Start

### 1. Fetch Market Data
```python
from data.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Fetch NIFTY 50 stocks data
pipeline.fetch_nifty50_data(start_date='2020-01-01', end_date='2024-12-31')

# Fetch option chain data
pipeline.fetch_option_chain('NIFTY', expiry_date='2024-01-25')
```

### 2. Screen for Undervalued Stocks
```python
from screening.equity_screener import EquityScreener

screener = EquityScreener()
results = screener.screen_undervalued_stocks(
    min_roe=15,
    max_pe=20,
    max_debt_equity=1.0,
    rsi_range=(30, 70)
)
print(results)
```

### 3. Backtest a Strategy
```python
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.backtest_engine import BacktestEngine

# Define strategy
strategy = MeanReversionStrategy(window=20, std_dev=2)

# Run backtest
engine = BacktestEngine(initial_capital=100000)
results = engine.run(strategy, symbol='RELIANCE', start='2020-01-01', end='2023-12-31')

# View performance
results.plot_performance()
print(results.get_metrics())
```

### 4. Analyze Options
```python
from options.volatility_analyzer import VolatilityAnalyzer

analyzer = VolatilityAnalyzer()
vol_surface = analyzer.calculate_volatility_surface('NIFTY')
analyzer.plot_volatility_smile()
```

### 5. Forecast Prices with ML
```python
from ml.price_forecaster import PriceForecaster

forecaster = PriceForecaster(model_type='lstm')
forecaster.train(symbol='TCS', lookback_days=60)
predictions = forecaster.predict(horizon_days=5)
forecaster.plot_predictions()
```

## 📁 Project Structure

```
qUANTUM/
├── data/                      # Data pipeline and storage
│   ├── data_pipeline.py       # Main data fetching engine
│   ├── data_cleaner.py        # Data validation and cleaning
│   └── raw/                   # Raw data storage
│
├── strategies/                # Trading strategies
│   ├── mean_reversion.py      # Mean reversion strategy
│   ├── momentum.py            # Momentum strategy
│   ├── pairs_trading.py       # Pairs trading strategy
│   └── options_strategies.py  # Options strategies
│
├── backtesting/               # Backtesting framework
│   ├── backtest_engine.py     # Main backtesting engine
│   ├── performance.py         # Performance metrics
│   └── risk_metrics.py        # Risk analysis
│
├── screening/                 # Stock screening tools
│   ├── equity_screener.py     # Fundamental + technical screener
│   ├── fundamental.py         # Fundamental analysis
│   └── technical.py           # Technical indicators
│
├── options/                   # Options analysis
│   ├── greeks_calculator.py   # Calculate option Greeks
│   ├── volatility_analyzer.py # Volatility modeling
│   └── option_chain.py        # Option chain analysis
│
├── ml/                        # Machine learning models
│   ├── price_forecaster.py    # Price prediction models
│   ├── features.py            # Feature engineering
│   └── models/                # Trained model storage
│
├── utils/                     # Utility functions
│   ├── indicators.py          # Technical indicators
│   ├── metrics.py             # Performance metrics
│   └── visualization.py       # Plotting utilities
│
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── symbols.yaml           # Stock symbols list
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_strategy_development.ipynb
│   ├── 03_backtesting_demo.ipynb
│   └── 04_ml_forecasting.ipynb
│
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📊 Supported Strategies

1. **Mean Reversion** - Bollinger Bands, RSI-based
2. **Momentum** - Moving average crossover, breakout strategies
3. **Pairs Trading** - Statistical arbitrage between correlated stocks
4. **Option Strategies** - Straddle, strangle, iron condor, bull/bear spreads

## 🎯 Performance Metrics

- **Returns**: Total return, CAGR, monthly/yearly returns
- **Risk**: Volatility, max drawdown, downside deviation
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Analysis**: Win rate, profit factor, average trade duration

## 🔬 Machine Learning Models

- **Linear Regression**: Trend analysis and support/resistance
- **Random Forest**: Feature importance and classification
- **LSTM Neural Networks**: Sequential price prediction
- **ARIMA/Prophet**: Time-series forecasting
- **XGBoost**: High-performance gradient boosting

## 📈 Data Sources

- Yahoo Finance (yfinance)
- NSE India (nsepy)
- Option chain data APIs
- Custom data providers (configurable)

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_backtesting.py

# Run with coverage
python -m pytest --cov=. tests/
```

## 📚 Documentation

Detailed documentation for each module is available in the `docs/` directory:
- [Data Pipeline Guide](docs/data_pipeline.md)
- [Strategy Development](docs/strategies.md)
- [Backtesting Framework](docs/backtesting.md)
- [ML Models](docs/machine_learning.md)

## ⚠️ Disclaimer

This project is for educational and research purposes only. Trading stocks and derivatives involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NSE India for market data
- Yahoo Finance for historical data
- Open-source Python community for amazing libraries

---

**Built with ❤️ for quantitative traders and researchers**
