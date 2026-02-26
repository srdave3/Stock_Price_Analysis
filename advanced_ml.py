"""
Advanced Machine Learning Models for Stock Prediction
XGBoost, Ensemble Methods, and Sentiment Analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import NEWS_API_KEY

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis from news and social media"""
    
    def __init__(self, api_key=None):
        """
        Initialize sentiment analyzer
        
        Args:
            api_key: News API key (from newsapi.org)
        """
        self.api_key = api_key or NEWS_API_KEY
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text):
        """
        Analyze sentiment of text using VADER
        
        Args:
            text: Text string
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        blob_sentiment = blob.sentiment
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': blob_sentiment.polarity,
            'textblob_subjectivity': blob_sentiment.subjectivity,
            'combined': (vader_scores['compound'] + blob_sentiment.polarity) / 2
        }
    
    def fetch_news(self, symbol, days=7):
        """
        Fetch news articles for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No API key for news sentiment")
            return []
        
        try:
            # Get company name from symbol (simplified - you'd want a proper mapping)
            company_map = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google',
                'JPM': 'JPMorgan',
                'JNJ': 'Johnson & Johnson',
                'TSLA': 'Tesla',
                'AMZN': 'Amazon',
                'META': 'Meta'
            }
            
            company = company_map.get(symbol, symbol)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # NewsAPI endpoint
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f'"{company}" OR "{symbol}"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                logger.info(f"Fetched {len(articles)} articles for {symbol}")
                return articles
            else:
                logger.error(f"News API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_sentiment_scores(self, symbol, days=7):
        """
        Get sentiment scores for a symbol over time
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
            
        Returns:
            DataFrame with daily sentiment scores
        """
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            return pd.DataFrame()
        
        # Process articles by date
        daily_scores = {}
        
        for article in articles:
            try:
                # Extract date
                pub_date = article['publishedAt'][:10]  # YYYY-MM-DD
                
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Get sentiment
                sentiment = self.analyze_text(text)
                
                # Store by date
                if pub_date not in daily_scores:
                    daily_scores[pub_date] = []
                
                daily_scores[pub_date].append(sentiment['combined'])
                
            except Exception as e:
                continue
        
        # Average scores by date
        sentiment_df = pd.DataFrame([
            {'date': date, 'sentiment': np.mean(scores)}
            for date, scores in daily_scores.items()
        ])
        
        if not sentiment_df.empty:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.set_index('date').sort_index()
        
        return sentiment_df


class AdvancedStockPredictor:
    """Advanced ML models for stock prediction"""
    
    def __init__(self, df, target_col='close', feature_cols=None):
        """
        Initialize predictor
        
        Args:
            df: DataFrame with stock data
            target_col: Target column to predict
            feature_cols: List of feature columns
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols or self._get_default_features()
        self.models = {}
        self.scaler = StandardScaler()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def _get_default_features(self):
        """Get default feature columns"""
        default_features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'rsi', 'macd',
            'bb_upper', 'bb_lower', 'atr'
        ]
        
        # Only keep columns that exist
        return [col for col in default_features if col in self.df.columns]
    
    def prepare_features(self, include_sentiment=True):
        """
        Prepare features for modeling
        
        Args:
            include_sentiment: Whether to include sentiment features
            
        Returns:
            X, y for modeling
        """
        df = self.df.copy()
        
        # Create additional features
        df = self._create_technical_features(df)
        df = self._create_lag_features(df)
        
        if include_sentiment:
            df = self._add_sentiment_features(df)
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Select features
        feature_cols = self.feature_cols.copy()
        
        # Add engineered features if they exist
        engineered = ['returns', 'volatility', 'volume_ratio', 'price_position']
        for col in engineered:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add lag features
        lag_cols = [col for col in df.columns if 'lag_' in col]
        feature_cols.extend(lag_cols)
        
        # Add sentiment features
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        feature_cols.extend(sentiment_cols)
        
        # Remove duplicates
        feature_cols = list(set(feature_cols))
        
        # Create target (next day's price)
        df['target'] = df[self.target_col].shift(-1)
        
        # Remove rows with NaN target
        df = df.dropna()
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y, feature_cols
    
    def _create_technical_features(self, df):
        """Create technical features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Volatility (rolling std of returns)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Price position in range
        df['price_range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / df['price_range']
        
        return df
    
    def _create_lag_features(self, df, lags=[1, 2, 3, 5, 10]):
        """Create lag features"""
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            if 'rsi' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            if 'macd' in df.columns:
                df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
        
        return df
    
    def _add_sentiment_features(self, df):
        """Add sentiment features"""
        try:
            # Get sentiment scores
            symbol = 'AAPL'  # You'd need to pass this properly
            sentiment_df = self.sentiment_analyzer.get_sentiment_scores(symbol, days=30)
            
            if not sentiment_df.empty:
                # Align dates
                for date, row in sentiment_df.iterrows():
                    if date in df.index:
                        df.loc[date, 'sentiment'] = row['sentiment']
                
                # Forward fill missing sentiment
                df['sentiment'] = df['sentiment'].fillna(method='ffill')
                df['sentiment'] = df['sentiment'].fillna(0)
                
                # Sentiment moving average
                df['sentiment_ma'] = df['sentiment'].rolling(window=3).mean()
            else:
                df['sentiment'] = 0
                df['sentiment_ma'] = 0
                
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            df['sentiment'] = 0
            df['sentiment_ma'] = 0
        
        return df
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model...")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evals = [(dtrain, 'train')]
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'seed': 42
        }
        
        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.models['xgboost'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best RF params: {grid_search.best_params_}")
        
        self.models['random_forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_gradient_boosting(self, X_train, y_train):
        """
        Train Gradient Boosting model
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Trained model
        """
        logger.info("Training Gradient Boosting model...")
        
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = gb
        return gb
    
    def train_ensemble(self, X_train, y_train):
        """
        Train ensemble model (voting regressor)
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Trained ensemble model
        """
        logger.info("Training Ensemble model...")
        
        # Train individual models if not already trained
        if 'xgboost' not in self.models:
            self.train_xgboost(X_train, y_train)
        
        if 'random_forest' not in self.models:
            self.train_random_forest(X_train, y_train)
        
        if 'gradient_boosting' not in self.models:
            self.train_gradient_boosting(X_train, y_train)
        
        # Create ensemble
        ensemble = VotingRegressor([
            ('xgb', self.models['xgboost'] if hasattr(self.models['xgboost'], 'predict') else 
                   xgb.XGBRegressor(**self.models['xgboost'].get_params())),
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting'])
        ])
        
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def train_all_models(self, test_size=0.2):
        """
        Train all models
        
        Args:
            test_size: Test set size
            
        Returns:
            Dictionary with model results
        """
        # Prepare data
        X, y, features = self.prepare_features()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data (preserve time order)
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        results = {}
        
        # Train each model
        models_to_train = {
            'xgboost': self.train_xgboost,
            'random_forest': self.train_random_forest,
            'gradient_boosting': self.train_gradient_boosting,
            'ensemble': self.train_ensemble
        }
        
        for name, train_func in models_to_train.items():
            try:
                model = train_func(X_train, y_train)
                
                # Predictions
                if name == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    y_pred = model.predict(dtest)
                else:
                    y_pred = model.predict(X_test)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"{name} - RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Store test data
        results['test_data'] = {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_values': y_test.values
        }
        
        return results
    
    def predict_future(self, days=30):
        """
        Predict future prices
        
        Args:
            days: Number of days to predict
            
        Returns:
            DataFrame with predictions
        """
        # Prepare latest data
        X, _, _ = self.prepare_features()
        latest_data = X.iloc[-1:].copy()
        
        predictions = []
        current_data = latest_data.copy()
        
        for i in range(days):
            # Scale
            current_scaled = pd.DataFrame(
                self.scaler.transform(current_data),
                columns=current_data.columns
            )
            
            # Predict with each model
            day_pred = {'day': i+1}
            
            for name, model in self.models.items():
                if name == 'xgboost':
                    dmatrix = xgb.DMatrix(current_scaled)
                    pred = model.predict(dmatrix)[0]
                elif hasattr(model, 'predict'):
                    pred = model.predict(current_scaled)[0]
                else:
                    continue
                
                day_pred[name] = pred
            
            predictions.append(day_pred)
            
            # Update for next prediction (simplified)
            # In production, you'd need to update all features
        
        return pd.DataFrame(predictions)
    
    def save_models(self, path='models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name != 'test_data':
                joblib.dump(model, f"{path}/{name}_model.pkl")
        
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path='models/'):
        """Load trained models"""
        for name in ['xgboost', 'random_forest', 'gradient_boosting', 'ensemble']:
            model_path = f"{path}/{name}_model.pkl"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
        
        scaler_path = f"{path}/scaler.pkl"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)


if __name__ == "__main__":
    # Test advanced ML
    print("Testing Advanced ML Models...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.randn(len(dates)),
        'high': prices + np.abs(np.random.randn(len(dates))),
        'low': prices - np.abs(np.random.randn(len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'sma_20': prices.rolling(20).mean(),
        'sma_50': prices.rolling(50).mean(),
        'ema_12': prices.ewm(span=12).mean(),
        'rsi': 50 + np.random.randn(len(dates)) * 10,
        'macd': np.random.randn(len(dates)) * 2,
        'bb_upper': prices * 1.02,
        'bb_lower': prices * 0.98,
        'atr': np.random.rand(len(dates)) * 2
    }, index=dates)
    
    print("\n1. Testing sentiment analysis:")
    sentiment = SentimentAnalyzer()
    text = "Apple stock surges on strong earnings report"
    scores = sentiment.analyze_text(text)
    print(f"Text: '{text}'")
    print(f"Sentiment: {scores}")
    
    print("\n2. Testing ML models:")
    predictor = AdvancedStockPredictor(sample_data)
    results = predictor.train_all_models(test_size=0.2)
    
    print("\nModel Performance:")
    for name, result in results.items():
        if name not in ['test_data'] and 'error' not in result:
            print(f"{name}: RMSE={result['rmse']:.4f}, R2={result['r2']:.4f}")