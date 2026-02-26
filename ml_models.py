"""
Machine Learning Models Module for Stock Price Analysis
Implements LSTM, Random Forest, and ensemble methods for price prediction
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StockPricePredictor:
    """Class for building and evaluating ML models for stock prediction"""
    
    def __init__(self, data, target_col='close', feature_cols=None):
        """
        Initialize the stock price predictor
        
        Args:
            data: DataFrame with stock data
            target_col: Target column for prediction
            feature_cols: List of feature columns
        """
        self.data = data.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'volume']
        
        # Will be set during preparation
        self.X = None
        self.y = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        
    def create_features(self, lookback=20):
        """
        Create features for ML models including lagged values
        
        Args:
            lookback: Number of past days to use as features
            
        Returns:
            DataFrame with additional features
        """
        logger.info(f"Creating features with lookback={lookback}...")
        
        df = self.data.copy()
        
        # Create lagged features
        for col in ['close', 'volume']:
            for lag in range(1, lookback + 1):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Return features
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Drop NaN rows
        df = df.dropna()
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def prepare_data(self, lookback=20, test_size=0.2):
        """
        Prepare data for ML training
        
        Args:
            lookback: Number of past days to use
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Create features
        df = self.create_features(lookback)
        
        # Select features (exclude target and non-feature columns)
        exclude_cols = [self.target_col, 'symbol']
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and col in self.feature_cols or '_lag_' in col or 'rolling_' in col 
                       or 'change' in col or 'momentum' in col or 'return' in col]
        
        # Use all numeric columns as features except target
        feature_cols = [col for col in df.columns if col != self.target_col and col != 'symbol']
        
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X = X_train_scaled
        self.y = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.feature_names = feature_cols
        
        logger.info(f"Data prepared: Train size={len(X_train)}, Test size={len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_regression(self):
        """
        Train Linear Regression model
        
        Returns:
            Trained model
        """
        logger.info("Training Linear Regression model...")
        
        if self.X is None:
            self.prepare_data()
        
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        self.models['linear_regression'] = model
        
        # Predictions
        y_pred = model.predict(self.X_test)
        self.predictions['linear_regression'] = y_pred
        
        # Evaluate
        metrics = self.evaluate_model(y_pred, 'Linear Regression')
        
        return model, metrics
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """
        Train Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Trained model
        """
        logger.info(f"Training Random Forest model (n={n_estimators}, depth={max_depth})...")
        
        if self.X is None:
            self.prepare_data()
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X, self.y)
        
        self.models['random_forest'] = model
        
        # Predictions
        y_pred = model.predict(self.X_test)
        self.predictions['random_forest'] = y_pred
        
        # Evaluate
        metrics = self.evaluate_model(y_pred, 'Random Forest')
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 important features:\n{importance.head(10)}")
        
        return model, metrics
    
    def train_gradient_boosting(self, n_estimators=100, max_depth=5):
        """
        Train Gradient Boosting model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            
        Returns:
            Trained model
        """
        logger.info(f"Training Gradient Boosting model (n={n_estimators}, depth={max_depth})...")
        
        if self.X is None:
            self.prepare_data()
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(self.X, self.y)
        
        self.models['gradient_boosting'] = model
        
        # Predictions
        y_pred = model.predict(self.X_test)
        self.predictions['gradient_boosting'] = y_pred
        
        # Evaluate
        metrics = self.evaluate_model(y_pred, 'Gradient Boosting')
        
        return model, metrics
    
    def train_all_models(self):
        """
        Train all available models
        
        Returns:
            Dictionary of trained models and their metrics
        """
        logger.info("Training all models...")
        
        results = {}
        
        # Linear Regression
        lr_model, lr_metrics = self.train_linear_regression()
        results['linear_regression'] = {'model': lr_model, 'metrics': lr_metrics}
        
        # Random Forest
        rf_model, rf_metrics = self.train_random_forest()
        results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
        
        # Gradient Boosting
        gb_model, gb_metrics = self.train_gradient_boosting()
        results['gradient_boosting'] = {'model': gb_model, 'metrics': gb_metrics}
        
        # Ensemble (average)
        self.create_ensemble()
        
        return results
    
    def create_ensemble(self, weights=None):
        """
        Create ensemble prediction from multiple models
        
        Args:
            weights: Dictionary of model weights
            
        Returns:
            Ensemble predictions
        """
        logger.info("Creating ensemble model...")
        
        if not self.predictions:
            self.train_all_models()
        
        # Default: equal weights
        if weights is None:
            weights = {name: 1/len(self.predictions) for name in self.predictions.keys()}
        
        # Weighted average
        ensemble_pred = np.zeros_like(list(self.predictions.values())[0])
        
        for name, pred in self.predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        self.predictions['ensemble'] = ensemble_pred
        
        # Evaluate
        metrics = self.evaluate_model(ensemble_pred, 'Ensemble')
        
        return ensemble_pred, metrics
    
    def evaluate_model(self, y_pred, model_name):
        """
        Evaluate model performance
        
        Args:
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.y_test is None:
            logger.warning("No test data available")
            return None
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # MAPE
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(self.y_test))
        pred_direction = np.sign(np.diff(y_pred))
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"{model_name} Metrics:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R2: {r2:.4f}")
        logger.info(f"  Direction Accuracy: {direction_accuracy:.2f}%")
        
        return metrics
    
    def compare_models(self):
        """
        Compare all trained models
        
        Returns:
            DataFrame with comparison
        """
        if not self.predictions:
            self.train_all_models()
        
        comparison = []
        
        for name, pred in self.predictions.items():
            metrics = self.evaluate_model(pred, name)
            metrics['model'] = name
            comparison.append(metrics)
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df[['model', 'rmse', 'mae', 'mape', 'r2', 'direction_accuracy']]
        
        logger.info(f"\nModel Comparison:\n{comparison_df}")
        
        return comparison_df
    
    def backtest_strategy(self, initial_capital=10000):
        """
        Backtest trading strategy based on model predictions
        
        Args:
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        logger.info(f"Backtesting strategy with initial capital=${initial_capital}...")
        
        if not self.predictions:
            self.train_all_models()
        
        # Use ensemble predictions
        predictions = self.predictions.get('ensemble', list(self.predictions.values())[0])
        
        # Calculate returns
        actual_returns = np.diff(self.y_test) / self.y_test[:-1]
        pred_signals = np.sign(np.diff(predictions))
        
        # Strategy returns
        strategy_returns = actual_returns * pred_signals
        
        # Cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod() * initial_capital
        buy_hold_cumulative = (self.y_test / self.y_test[0]) * initial_capital
        
        # Performance metrics
        strategy_total_return = (strategy_cumulative[-1] - initial_capital) / initial_capital
        buy_hold_return = (buy_hold_cumulative[-1] - initial_capital) / initial_capital
        
        # Sharpe ratio (simplified)
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # Max drawdown
        strategy_cummax = np.maximum.accumulate(strategy_cumulative)
        drawdown = (strategy_cumulative - strategy_cummax) / strategy_cummax
        max_drawdown = np.min(drawdown)
        
        results = {
            'strategy_return': strategy_total_return,
            'buy_hold_return': buy_hold_return,
            'strategy_sharpe': strategy_sharpe,
            'max_drawdown': max_drawdown,
            'final_capital': strategy_cumulative[-1],
            'buy_hold_final': buy_hold_cumulative[-1],
            'cumulative_returns': strategy_cumulative,
            'buy_hold_returns': buy_hold_cumulative
        }
        
        logger.info(f"Strategy Return: {strategy_total_return:.2%}")
        logger.info(f"Buy & Hold Return: {buy_hold_return:.2%}")
        logger.info(f"Strategy Sharpe: {strategy_sharpe:.4f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        
        return results


def predict_stock(df, models_to_train=None):
    """
    Convenience function for stock prediction
    
    Args:
        df: Stock DataFrame
        models_to_train: List of models to train
        
    Returns:
        Predictions and metrics
    """
    predictor = StockPricePredictor(df)
    
    if models_to_train is None:
        predictor.train_all_models()
    else:
        for model_name in models_to_train:
            if model_name == 'linear_regression':
                predictor.train_linear_regression()
            elif model_name == 'random_forest':
                predictor.train_random_forest()
            elif model_name == 'gradient_boosting':
                predictor.train_gradient_boosting()
    
    return predictor.compare_models()


if __name__ == "__main__":
    # Test ML models
    print("Testing ML Models...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    
    # Generate random walk prices
    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.randn(n),
        'high': prices + np.abs(np.random.randn(n)),
        'low': prices - np.abs(np.random.randn(n)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    # Test prediction
    predictor = StockPricePredictor(sample_data)
    
    print("\n1. Preparing Data:")
    predictor.prepare_data(lookback=10, test_size=0.2)
    print(f"Features: {len(predictor.feature_names)}")
    
    print("\n2. Training Models:")
    results = predictor.train_all_models()
    
    print("\n3. Comparing Models:")
    comparison = predictor.compare_models()
    print(comparison)
    
    print("\n4. Backtesting Strategy:")
    backtest = predictor.backtest_strategy(initial_capital=10000)
    print(f"Strategy Return: {backtest['strategy_return']:.2%}")
    print(f"Buy & Hold Return: {backtest['buy_hold_return']:.2%}")
