"""
Price Forecaster
Machine learning models for short-term price movement prediction
Includes: Linear Regression, Random Forest, LSTM, ARIMA, XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from typing import Dict, Optional, Tuple, List
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PriceForecaster:
    """
    Forecast stock prices using various ML techniques
    """
    
    def __init__(self, model_type: str = 'lstm'):
        """
        Initialize price forecaster
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'random_forest', 'xgboost', 'lstm', 'arima'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.trained = False
        
        # LSTM specific
        self.lookback_days = 60
        self.sequence_data = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with engineered features
        """
        from .features import FeatureEngineer
        
        engineer = FeatureEngineer()
        features_df = engineer.create_all_features(df)
        
        return features_df
    
    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data
            lookback: Number of time steps to look back
        
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Predict close price
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        test_size: float = 0.2,
        **model_params
    ) -> Dict:
        """
        Train the forecasting model
        
        Args:
            data: Training data with OHLCV
            target_col: Target column to predict
            test_size: Test set size
            **model_params: Additional model parameters
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare features
        features_df = self.prepare_features(data)
        features_df = features_df.dropna()
        
        if len(features_df) == 0:
            logger.error("No valid data after feature engineering")
            return {}
        
        # Select features (exclude target and date columns)
        exclude_cols = [target_col, 'Date'] if 'Date' in features_df.columns else [target_col]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.feature_columns = feature_cols
        
        # Train based on model type
        if self.model_type == 'linear':
            metrics = self._train_linear(X_train, y_train, X_test, y_test)
        elif self.model_type == 'ridge':
            metrics = self._train_ridge(X_train, y_train, X_test, y_test, **model_params)
        elif self.model_type == 'lasso':
            metrics = self._train_lasso(X_train, y_train, X_test, y_test, **model_params)
        elif self.model_type == 'random_forest':
            metrics = self._train_random_forest(X_train, y_train, X_test, y_test, **model_params)
        elif self.model_type == 'xgboost':
            metrics = self._train_xgboost(X_train, y_train, X_test, y_test, **model_params)
        elif self.model_type == 'lstm':
            metrics = self._train_lstm(X_train, y_train, X_test, y_test, **model_params)
        elif self.model_type == 'arima':
            metrics = self._train_arima(y_train, y_test, **model_params)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return {}
        
        self.trained = True
        logger.info("Training complete!")
        
        return metrics
    
    def _train_linear(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train linear regression model"""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return self._calculate_metrics(y_test, y_pred)
    
    def _train_ridge(self, X_train, y_train, X_test, y_test, alpha=1.0) -> Dict:
        """Train ridge regression model"""
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return self._calculate_metrics(y_test, y_pred)
    
    def _train_lasso(self, X_train, y_train, X_test, y_test, alpha=1.0) -> Dict:
        """Train lasso regression model"""
        self.model = Lasso(alpha=alpha)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return self._calculate_metrics(y_test, y_pred)
    
    def _train_random_forest(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=100,
        max_depth=10
    ) -> Dict:
        """Train random forest model"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.head(10).to_dict('records')
        
        return metrics
    
    def _train_xgboost(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    ) -> Dict:
        """Train XGBoost model"""
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.head(10).to_dict('records')
        
        return metrics
    
    def _train_lstm(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=32,
        units=[50, 50],
        dropout=0.2
    ) -> Dict:
        """Train LSTM model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Scale data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(
                np.column_stack([y_train.reshape(-1, 1), X_train_scaled]),
                self.lookback_days
            )
            X_test_seq, y_test_seq = self.create_sequences(
                np.column_stack([y_test.reshape(-1, 1), X_test_scaled]),
                self.lookback_days
            )
            
            # Build LSTM model
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(units[0], return_sequences=True if len(units) > 1 else False,
                          input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(Dropout(dropout))
            
            # Additional LSTM layers
            for i in range(1, len(units)):
                return_seq = i < len(units) - 1
                model.add(LSTM(units[i], return_sequences=return_seq))
                model.add(Dropout(dropout))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            # Predict
            y_pred = model.predict(X_test_seq, verbose=0)
            
            self.model = model
            
            metrics = self._calculate_metrics(y_test_seq, y_pred.flatten())
            metrics['final_train_loss'] = float(history.history['loss'][-1])
            metrics['final_val_loss'] = float(history.history['val_loss'][-1])
            
            return metrics
        
        except ImportError:
            logger.error("TensorFlow not installed, cannot train LSTM")
            return {}
    
    def _train_arima(self, y_train, y_test, order=(5, 1, 0)) -> Dict:
        """Train ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Fit ARIMA
            model = ARIMA(y_train, order=order)
            self.model = model.fit()
            
            # Forecast
            forecast = self.model.forecast(steps=len(y_test))
            
            return self._calculate_metrics(y_test, forecast)
        
        except ImportError:
            logger.error("statsmodels not installed, cannot train ARIMA")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """Calculate prediction metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0.0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy)
        }
    
    def predict(self, data: pd.DataFrame, horizon_days: int = 5) -> np.ndarray:
        """
        Make price predictions
        
        Args:
            data: Input data
            horizon_days: Number of days to forecast
        
        Returns:
            Array of predictions
        """
        if not self.trained:
            logger.error("Model not trained yet")
            return np.array([])
        
        features_df = self.prepare_features(data)
        features_df = features_df.dropna()
        
        if len(features_df) == 0:
            return np.array([])
        
        X = features_df[self.feature_columns].values
        
        if self.model_type == 'lstm':
            X_scaled = self.scaler.transform(X)
            # Use last lookback_days for prediction
            X_seq = X_scaled[-self.lookback_days:].reshape(1, self.lookback_days, X_scaled.shape[1])
            predictions = self.model.predict(X_seq, verbose=0)
        elif self.model_type == 'arima':
            predictions = self.model.forecast(steps=horizon_days)
        else:
            # For other models, predict on last data point
            predictions = self.model.predict(X[-horizon_days:])
        
        return predictions.flatten()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.trained:
            logger.warning("No trained model to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'lstm':
            self.model.save(filepath)
        else:
            joblib.dump(self.model, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.model_type == 'lstm':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        
        self.trained = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data.data_pipeline import DataPipeline
    
    pipeline = DataPipeline()
    data = pipeline.fetch_stock_data('TCS.NS', start_date='2022-01-01')
    
    # Linear Regression
    forecaster = PriceForecaster(model_type='linear')
    metrics = forecaster.train(data)
    
    print("Linear Regression Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    
    # Random Forest
    forecaster_rf = PriceForecaster(model_type='random_forest')
    metrics_rf = forecaster_rf.train(data, n_estimators=100)
    
    print("\nRandom Forest Metrics:")
    print(f"RMSE: {metrics_rf['rmse']:.2f}")
    print(f"R²: {metrics_rf['r2']:.4f}")
    print(f"Direction Accuracy: {metrics_rf['direction_accuracy']:.2f}%")

