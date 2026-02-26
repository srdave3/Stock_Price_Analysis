"""
Configuration file for Stock Price Analysis Project
Enhanced version with dashboard, real-time monitoring, and advanced ML features
"""

import os
from datetime import datetime, timedelta

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EXPORTS_DIR = os.path.join(DATA_DIR, 'exports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

# Create directories if they don't exist
for directory in [DATA_DIR, EXPORTS_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR, 
                  REPORTS_DIR, TEMPLATES_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# STOCK CONFIGURATION
# ============================================================================

STOCKS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'symbol': 'AAPL',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'apple_logo.png'
    },
    'JPM': {
        'name': 'JPMorgan Chase & Co.',
        'sector': 'Finance',
        'symbol': 'JPM',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'jpm_logo.png'
    },
    'JNJ': {
        'name': 'Johnson & Johnson',
        'sector': 'Healthcare',
        'symbol': 'JNJ',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'jnj_logo.png'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'sector': 'Technology',
        'symbol': 'MSFT',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'msft_logo.png'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'symbol': 'GOOGL',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'google_logo.png'
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'sector': 'Automotive',
        'symbol': 'TSLA',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'tesla_logo.png'
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'sector': 'E-commerce',
        'symbol': 'AMZN',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'amazon_logo.png'
    },
    'META': {
        'name': 'Meta Platforms Inc.',
        'sector': 'Technology',
        'symbol': 'META',
        'country': 'USA',
        'currency': 'USD',
        'logo': 'meta_logo.png'
    }
}

# Market Index
MARKET_INDEX = '^GSPC'  # S&P 500
MARKET_INDEX_NAME = 'S&P 500'

# Additional indices for comparison
ADDITIONAL_INDICES = {
    '^IXIC': 'NASDAQ Composite',
    '^DJI': 'Dow Jones Industrial Average',
    '^RUT': 'Russell 2000',
    '^VIX': 'CBOE Volatility Index'
}

# ============================================================================
# DATE RANGE
# ============================================================================

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)  # 5 years of data

# For real-time updates
REALTIME_UPDATE_INTERVAL = 60  # seconds
HISTORICAL_UPDATE_INTERVAL = 300  # seconds (5 minutes)

# ============================================================================
# TECHNICAL ANALYSIS PARAMETERS
# ============================================================================

TECHNICAL_PARAMS = {
    # Moving Averages
    'sma_short': 20,
    'sma_medium': 50,
    'sma_long': 200,
    'ema_short': 12,
    'ema_medium': 26,
    'ema_long': 50,
    
    # RSI
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    
    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # Bollinger Bands
    'bollinger_period': 20,
    'bollinger_std': 2,
    
    # Stochastic
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'stoch_slow_period': 3,
    
    # ADX
    'adx_period': 14,
    
    # ATR
    'atr_period': 14,
    
    # CCI
    'cci_period': 20,
    
    # MFI
    'mfi_period': 14,
    
    # Williams %R
    'williams_period': 14,
    
    # Ichimoku Cloud
    'ichimoku_conversion': 9,
    'ichimoku_base': 26,
    'ichimoku_span': 52
}

# ============================================================================
# RISK METRICS PARAMETERS
# ============================================================================

# Risk-Free Rate (US Treasury 10-year yield approximation)
RISK_FREE_RATE = 0.05  # 5% annual

# Confidence levels for VaR/CVaR
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Trading days per year
TRADING_DAYS = 252

# ============================================================================
# PORTFOLIO OPTIMIZATION PARAMETERS
# ============================================================================

PORTFOLIO_PARAMS = {
    'min_weight': 0.05,        # Minimum 5% per asset
    'max_weight': 0.40,        # Maximum 40% per asset
    'target_return': None,      # None for max Sharpe ratio
    'risk_aversion': 2.0,       # Risk aversion coefficient
    'n_random_portfolios': 10000,  # For Monte Carlo simulation
    'optimization_method': 'SLSQP',  # Optimization algorithm
    'rebalance_frequency': 'quarterly'  # 'monthly', 'quarterly', 'yearly'
}

# ============================================================================
# REAL-TIME MONITORING PARAMETERS
# ============================================================================

ALERT_THRESHOLDS = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'volume_spike': 2.0,           # 2x average volume
    'price_change_percent': 5.0,     # 5% price change
    'macd_cross': True,
    'golden_cross': True,
    'death_cross': True,
    'bb_breakout': True,
    'support_resistance_break': 0.02,  # 2% break
    'consecutive_gains': 5,
    'consecutive_losses': 5
}

# Alert notification methods
ALERT_METHODS = {
    'desktop': True,
    'email': False,  # Set to True and configure below
    'sms': False,    # Set to True and configure below
    'webhook': False,
    'telegram': False
}

# Email configuration (if enabled)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'from_email': 'your_email@gmail.com',
    'from_password': 'your_app_password',
    'to_email': 'alerts@yourdomain.com'
}

# Telegram configuration (if enabled)
TELEGRAM_CONFIG = {
    'bot_token': 'YOUR_BOT_TOKEN',
    'chat_id': 'YOUR_CHAT_ID'
}

# Webhook configuration (if enabled)
WEBHOOK_URL = 'https://your-webhook-url.com/endpoint'

# ============================================================================
# ADVANCED ML PARAMETERS
# ============================================================================

ML_PARAMS = {
    # General
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'cv_folds': 5,
    
    # LSTM
    'lstm_units': [50, 50],
    'lstm_dropout': 0.2,
    'lstm_recurrent_dropout': 0.2,
    'lstm_epochs': 100,
    'lstm_batch_size': 32,
    'lstm_early_stopping_patience': 10,
    
    # Random Forest
    'rf_n_estimators': 200,
    'rf_max_depth': 15,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    'rf_max_features': 'sqrt',
    
    # XGBoost
    'xgb_n_estimators': 200,
    'xgb_max_depth': 8,
    'xgb_learning_rate': 0.05,
    'xgb_subsample': 0.8,
    'xgb_colsample_bytree': 0.8,
    'xgb_reg_alpha': 0.1,
    'xgb_reg_lambda': 0.1,
    
    # Gradient Boosting
    'gb_n_estimators': 200,
    'gb_max_depth': 6,
    'gb_learning_rate': 0.05,
    'gb_subsample': 0.8,
    
    # Ensemble
    'ensemble_weights': [0.3, 0.3, 0.2, 0.2],  # RF, XGB, GB, LSTM
    
    # Feature Engineering
    'use_sentiment': True,
    'use_technical_features': True,
    'use_lag_features': True,
    'lag_periods': [1, 2, 3, 5, 10, 20],
    'rolling_windows': [5, 10, 20, 50],
    
    # Training
    'early_stopping_rounds': 50,
    'n_iter_search': 20,  # For randomized search
    'scoring_metric': 'neg_mean_squared_error'
}

# ============================================================================
# SENTIMENT ANALYSIS PARAMETERS
# ============================================================================

# News API (get free key from newsapi.org)
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')

# Alpha Vantage API (get free key from alphavantage.co)
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '')

# Twitter API (optional)
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.environ.get('TWITTER_API_SECRET', '')

# Sentiment analysis settings
SENTIMENT_PARAMS = {
    'news_days_lookback': 7,
    'min_articles_per_day': 3,
    'sentiment_sources': ['news', 'twitter'],  # 'news', 'twitter', 'reddit'
    'use_vader': True,
    'use_textblob': True,
    'combine_method': 'average'  # 'average', 'weighted', 'max'
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================

# Flask settings
FLASK_CONFIG = {
    'SECRET_KEY': 'stock-analysis-secret-key-change-this-in-production',
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'USE_RELOADER': True
}

# Socket.IO settings
SOCKETIO_CONFIG = {
    'cors_allowed_origins': '*',
    'async_mode': 'threading',
    'ping_timeout': 60,
    'ping_interval': 25
}

# Dashboard themes
AVAILABLE_THEMES = ['dark', 'light', 'cyberpunk', 'corporate']
DEFAULT_THEME = 'dark'

# Chart settings
CHART_CONFIG = {
    'default_height': 600,
    'default_width': 1000,
    'template': 'plotly_dark',
    'show_grid': True,
    'animation_duration': 500
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot styles
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
DPI = 100

# Color schemes
COLOR_SCHEMES = {
    'primary': '#007bff',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'background': '#1a1a1a',
    'text': '#ffffff'
}

# Technical indicator colors
INDICATOR_COLORS = {
    'sma': 'blue',
    'ema': 'orange',
    'rsi': 'purple',
    'macd': 'green',
    'macd_signal': 'red',
    'bb_upper': 'gray',
    'bb_lower': 'gray',
    'volume': 'lightblue',
    'obv': 'darkblue'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, f'stock_analysis_{datetime.now().strftime("%Y%m%d")}.log')

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    'use_cache': True,
    'cache_expiry_days': 1,
    'max_cache_size_mb': 500,
    'cache_compression': True
}

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

EXPORT_FORMATS = ['csv', 'excel', 'json', 'html']
DEFAULT_EXPORT_FORMAT = 'csv'
INCLUDE_TIMESTAMP_IN_EXPORTS = True

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Rate limiting
API_RATE_LIMIT = 5  # requests per second
API_TIMEOUT = 30  # seconds

# Yahoo Finance settings
YFINANCE_CONFIG = {
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Parallel processing
USE_MULTIPROCESSING = True
MAX_WORKERS = 4

# Data sampling
MAX_DATA_POINTS_FOR_DISPLAY = 1000
DATA_DOWNSAMPLING_THRESHOLD = 10000

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_FEATURES = {
    'real_time_monitoring': True,
    'portfolio_optimization': True,
    'ml_predictions': True,
    'sentiment_analysis': False,  # Set to True if you have API keys
    'email_alerts': False,
    'telegram_alerts': False,
    'data_caching': True,
    'parallel_processing': True,
    'auto_update': True
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    assert RISK_FREE_RATE >= 0 and RISK_FREE_RATE <= 1, "Risk-free rate must be between 0 and 1"
    assert TRADING_DAYS in [252, 365], "Trading days should be 252 or 365"
    assert PORTFOLIO_PARAMS['min_weight'] <= PORTFOLIO_PARAMS['max_weight'], "Min weight must be <= max weight"
    
    # Check if API keys are set when features are enabled
    if ENABLE_FEATURES['sentiment_analysis'] and not NEWS_API_KEY:
        print("Warning: Sentiment analysis enabled but NEWS_API_KEY not set")
    
    print("✓ Configuration validation passed")

# Run validation on import
if __name__ != '__main__':
    validate_config()
