"""
Data Preparation Module for Stock Price Analysis
Cleans, preprocesses, and prepares stock data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class DataPreparator:
    """Class to clean and prepare stock data"""
    
    def __init__(self, data=None):
        """
        Initialize the data preparator
        
        Args:
            data: DataFrame or dictionary of DataFrames
        """
        self.data = data
        self.processed_data = {}
        
    def load_data(self, filepath, symbol=None):
        """
        Load stock data from CSV
        
        Args:
            filepath: Path to CSV file
            symbol: Stock symbol (optional, for dict storage)
            
        Returns:
            DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
        
        if symbol:
            self.data = {symbol: df}
        else:
            self.data = df
            
        return df
    
    def check_missing_values(self, df):
        """
        Check for missing values in DataFrame
        
        Args:
            df: DataFrame to check
            
        Returns:
            DataFrame with missing value counts
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        
        logger.info(f"Missing values:\n{missing_df}")
        return missing_df
    
    def handle_missing_values(self, df, method='forward'):
        """
        Handle missing values in the data
        
        Args:
            df: DataFrame to process
            method: Method to handle missing values
                   - 'forward': Forward fill
                   - 'backward': Backward fill
                   - 'interpolate': Linear interpolation
                   - 'drop': Drop rows with missing values
                    
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using method: {method}")
        
        df_clean = df.copy()
        
        if method == 'forward':
            df_clean = df_clean.ffill()
        elif method == 'backward':
            df_clean = df_clean.bfill()
        elif method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
        elif method == 'drop':
            df_clean = df_clean.dropna()
        
        df_clean = df_clean.ffill().bfill()
        
        return df_clean
    
    def detect_outliers(self, df, column='close', threshold=3):
        """
        Detect outliers using z-score method
        
        Args:
            df: DataFrame
            column: Column to check for outliers
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers marked
        """
        df_with_outliers = df.copy()
        
        if column in df.columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df_with_outliers['is_outlier'] = z_scores > threshold
            
            outlier_count = df_with_outliers['is_outlier'].sum()
            logger.info(f"Found {outlier_count} outliers in {column}")
        
        return df_with_outliers
    
    def calculate_returns(self, df, columns=None):
        """
        Calculate daily and log returns
        
        Args:
            df: DataFrame with price data
            columns: Columns to calculate returns for
            
        Returns:
            DataFrame with return columns added
        """
        df_returns = df.copy()
        
        if columns is None:
            columns = ['close']
        
        for col in columns:
            if col in df.columns:
                df_returns[f'{col}_return'] = df[col].pct_change()
                df_returns[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        return df_returns
    
    def add_time_features(self, df):
        """
        Add time-based features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        df_features = df.copy()
        
        # Handle datetime index - fix timezone issues
        if not isinstance(df_features.index, pd.DatetimeIndex):
            df_features.index = pd.to_datetime(df_features.index, utc=True)
            df_features.index = df_features.index.tz_localize(None)
        elif df_features.index.tz is not None:
            df_features.index = df_features.index.tz_convert(None)
        
        # Time features
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['day'] = df_features.index.day
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_name'] = df_features.index.day_name()
        df_features['week_of_year'] = df_features.index.isocalendar().week
        df_features['quarter'] = df_features.index.quarter
        
        # Is weekend
        df_features['is_weekend'] = df_features.index.dayofweek.isin([5, 6])
        
        # Is month start/end
        df_features['is_month_start'] = df_features.index.is_month_start
        df_features['is_month_end'] = df_features.index.is_month_end
        
        return df_features
    
    def add_price_features(self, df):
        """
        Add price-based derived features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        
        # Price range
        df_features['high_low_range'] = df_features['high'] - df_features['low']
        
        # Price change
        df_features['close_open_diff'] = df_features['close'] - df_features['open']
        
        # Intraday return
        df_features['intraday_return'] = (df_features['close'] - df_features['open']) / df_features['open']
        
        # Gap
        df_features['gap'] = df_features['open'] - df_features['close'].shift(1)
        
        # VWAP
        df_features['vwap'] = (df_features['close'] * df_features['volume']).cumsum() / df_features['volume'].cumsum()
        
        # ATR
        df_features['tr'] = np.maximum(
            df_features['high'] - df_features['low'],
            np.maximum(
                np.abs(df_features['high'] - df_features['close'].shift(1)),
                np.abs(df_features['low'] - df_features['close'].shift(1))
            )
        )
        df_features['atr'] = df_features['tr'].rolling(window=14).mean()
        
        return df_features
    
    def normalize_data(self, df, columns=None, method='minmax'):
        """
        Normalize data
        
        Args:
            df: DataFrame
            columns: Columns to normalize
            method: 'minmax' or 'zscore'
            
        Returns:
            DataFrame with normalized columns
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = ['close']
        
        for col in columns:
            if col in df.columns:
                if method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df_norm[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    df_norm[f'{col}_zscore'] = (df[col] - mean) / std
        
        return df_norm
    
    def resample_data(self, df, freq='W'):
        """
        Resample data
        
        Args:
            df: DataFrame with datetime index
            freq: Target frequency
            
        Returns:
            Resampled DataFrame
        """
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        logger.info(f"Resampled data to {freq} frequency")
        return resampled
    
    def calculate_statistics(self, df, columns=None):
        """
        Calculate statistics
        
        Args:
            df: DataFrame
            columns: Columns to calculate statistics for
            
        Returns:
            DataFrame with statistics
        """
        if columns is None:
            columns = ['close']
        
        stats = {}
        
        for col in columns:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'variance': df[col].var()
                }
        
        stats_df = pd.DataFrame(stats).T
        logger.info(f"Statistical measures:\n{stats_df}")
        
        return stats_df
    
    def prepare_for_analysis(self, df):
        """
        Complete data preparation pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Fully prepared DataFrame
        """
        logger.info("Starting data preparation pipeline...")
        
        df_clean = self.handle_missing_values(df)
        df_clean = self.calculate_returns(df_clean)
        df_clean = self.add_time_features(df_clean)
        df_clean = self.add_price_features(df_clean)
        
        logger.info("Data preparation complete!")
        
        return df_clean
    
    def prepare_all_stocks(self, stock_data_dict):
        """
        Prepare all stocks
        
        Args:
            stock_data_dict: Dictionary of DataFrames
            
        Returns:
            Dictionary of prepared DataFrames
        """
        prepared = {}
        
        for symbol, df in stock_data_dict.items():
            logger.info(f"Preparing data for {symbol}")
            prepared[symbol] = self.prepare_for_analysis(df)
        
        return prepared


def prepare_stock_data(df):
    """Convenience function"""
    preparator = DataPreparator()
    return preparator.prepare_for_analysis(df)


if __name__ == "__main__":
    print("Testing Data Preparation...")
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates))),
        'high': 105 + np.cumsum(np.random.randn(len(dates))),
        'low': 95 + np.cumsum(np.random.randn(len(dates))),
        'close': 100 + np.cumsum(np.random.randn(len(dates))),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    sample_data.loc[sample_data.index[10], 'close'] = np.nan
    sample_data.loc[sample_data.index[50:55], 'volume'] = np.nan
    
    preparator = DataPreparator()
    
    print("\n1. Checking missing values:")
    preparator.check_missing_values(sample_data)
    
    print("\n2. Preparing data:")
    prepared = preparator.prepare_for_analysis(sample_data)
    print(f"Prepared data shape: {prepared.shape}")
    print(prepared.columns.tolist())
    
    print("\n3. Statistics:")
    stats = preparator.calculate_statistics(prepared)
    print(stats)
