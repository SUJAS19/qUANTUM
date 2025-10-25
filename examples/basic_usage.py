"""
Basic Usage Example for qUANTUM
Demonstrates core functionality of the framework
"""

import sys
sys.path.append('..')

from data.data_pipeline import DataPipeline
from screening.equity_screener import EquityScreener
from backtesting.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from ml.price_forecaster import PriceForecaster
from options.greeks_calculator import GreeksCalculator
from options.volatility_analyzer import VolatilityAnalyzer


def example_data_fetching():
    """Example 1: Fetch market data"""
    print("=" * 60)
    print("EXAMPLE 1: DATA FETCHING")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    # Fetch single stock
    print("\nFetching TCS data...")
    tcs_data = pipeline.fetch_stock_data('TCS.NS', start_date='2023-01-01')
    print(f"Fetched {len(tcs_data)} days of data")
    print("\nLatest data:")
    print(tcs_data.tail())
    
    # Fetch NIFTY 50 index
    print("\n\nFetching NIFTY 50 index...")
    nifty_data = pipeline.fetch_index_data('^NSEI', start_date='2023-01-01')
    print(f"Current NIFTY: {nifty_data['Close'].iloc[-1]:.2f}")


def example_screening():
    """Example 2: Screen for undervalued stocks"""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: STOCK SCREENING")
    print("=" * 60)
    
    screener = EquityScreener()
    
    print("\nScreening for undervalued stocks...")
    results = screener.screen_undervalued_stocks(
        min_roe=15,
        max_pe=25,
        max_debt_equity=1.0,
        rsi_range=(30, 70)
    )
    
    if not results.empty:
        print(f"\nFound {len(results)} stocks matching criteria:")
        print(results[['Symbol', 'Price', 'PE_Ratio', 'ROE', 'RSI', 'Value_Score']].head(10))
    else:
        print("No stocks found matching criteria")


def example_backtesting():
    """Example 3: Backtest a trading strategy"""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: BACKTESTING")
    print("=" * 60)
    
    # Fetch data
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
    
    print("\nBacktesting Mean Reversion Strategy on RELIANCE...")
    
    # Create strategy
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(strategy, data, symbol='RELIANCE.NS')
    
    # Print results
    results.print_summary()


def example_ml_forecasting():
    """Example 4: ML-based price forecasting"""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 4: ML PRICE FORECASTING")
    print("=" * 60)
    
    # Fetch data
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2022-01-01')
    
    print("\nTraining Random Forest model on TCS...")
    
    # Train model
    forecaster = PriceForecaster(model_type='random_forest')
    metrics = forecaster.train(data, test_size=0.2)
    
    print("\nModel Performance:")
    print(f"RMSE: ₹{metrics['rmse']:.2f}")
    print(f"MAE: ₹{metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    
    if 'feature_importance' in metrics:
        print("\nTop 5 Important Features:")
        for feat in metrics['feature_importance'][:5]:
            print(f"  {feat['feature']}: {feat['importance']:.4f}")


def example_options_analysis():
    """Example 5: Options Greeks calculation"""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 5: OPTIONS GREEKS")
    print("=" * 60)
    
    calc = GreeksCalculator()
    
    # NIFTY option parameters
    S = 19500  # Spot price
    K = 19500  # Strike (ATM)
    T = 30 / 365  # 30 days to expiration
    r = 0.07  # 7% risk-free rate
    sigma = 0.15  # 15% volatility
    
    print(f"\nCalculating Greeks for NIFTY {K} Call Option:")
    print(f"Spot: {S}, Strike: {K}, DTE: 30 days, IV: {sigma*100}%")
    
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    
    print(f"\nOption Price: ₹{greeks['price']:.2f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Theta: ₹{greeks['theta']:.4f} per day")
    print(f"Vega: ₹{greeks['vega']:.4f} per 1% IV change")
    print(f"Rho: ₹{greeks['rho']:.4f} per 1% rate change")


def example_volatility_analysis():
    """Example 6: Volatility analysis"""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 6: VOLATILITY ANALYSIS")
    print("=" * 60)
    
    # Fetch data
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('NIFTY50.NS', start_date='2023-01-01')
    
    analyzer = VolatilityAnalyzer()
    
    # Calculate historical volatility
    hist_vol = analyzer.calculate_historical_volatility(data['Close'])
    current_vol = hist_vol.iloc[-1]
    
    print(f"\nNIFTY 50 Volatility Analysis:")
    print(f"Current 30-day Historical Volatility: {current_vol*100:.2f}%")
    
    # Volatility regime
    regime = analyzer.analyze_volatility_regime(hist_vol.dropna(), current_vol)
    
    print(f"\nVolatility Regime: {regime['regime']}")
    print(f"Percentile: {regime['percentile']:.2f}%")
    print(f"Historical Mean: {regime['historical_mean']:.2f}%")
    print(f"Interpretation: {regime['interpretation']}")


def main():
    """Run all examples"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  qUANTUM - Quantitative Trading Framework Examples  ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    try:
        # Run examples
        example_data_fetching()
        example_screening()
        example_backtesting()
        example_ml_forecasting()
        example_options_analysis()
        example_volatility_analysis()
        
        print("\n\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

