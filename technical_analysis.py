"""
Technical Analysis Module for Stock Price Analysis
Implements technical indicators and chart patterns
"""

import pandas as pd
import numpy as np
import logging
from config import TECHNICAL_PARAMS

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Class for technical analysis of stock data"""
    
    def __init__(self, data, params=None):
        """
        Initialize the technical analyzer
        
        Args:
            data: DataFrame with OHLCV data
            params: Technical parameters dictionary
        """
        self.data = data.copy()
        self.params = params or TECHNICAL_PARAMS.copy()
        
    def calculate_sma(self, column='close', periods=None):
        """Calculate Simple Moving Average"""
        if periods is None:
            periods = [
                self.params.get('sma_short', 20),
                self.params.get('sma_medium', 50),
                self.params.get('sma_long', 200)
            ]
        
        df = self.data.copy()
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period, min_periods=1).mean()
        
        logger.info(f"Calculated SMA for periods: {periods}")
        return df
    
    def calculate_ema(self, column='close', periods=None):
        """Calculate Exponential Moving Average"""
        if periods is None:
            periods = [12, 26, 50]
        
        df = self.data.copy()
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        logger.info(f"Calculated EMA for periods: {periods}")
        return df
    
    def calculate_rsi(self, column='close', period=None):
        """Calculate Relative Strength Index (RSI)"""
        if period is None:
            period = self.params.get('rsi_period', 14)
        
        df = self.data.copy()
        delta = df[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Handle infinite values
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], 50)
        
        logger.info(f"Calculated RSI with period: {period}")
        return df
    
    def calculate_macd(self, column='close', fast=None, slow=None, signal=None):
        """Calculate MACD"""
        if fast is None:
            fast = self.params.get('macd_fast', 12)
        if slow is None:
            slow = self.params.get('macd_slow', 26)
        if signal is None:
            signal = self.params.get('macd_signal', 9)
        
        df = self.data.copy()
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        logger.info(f"Calculated MACD ({fast}, {slow}, {signal})")
        return df
    
    def calculate_bollinger_bands(self, column='close', period=None, std_dev=None):
        """Calculate Bollinger Bands"""
        if period is None:
            period = self.params.get('bollinger_period', 20)
        if std_dev is None:
            std_dev = self.params.get('bollinger_std', 2)
        
        df = self.data.copy()
        df['bb_middle'] = df[column].rolling(window=period, min_periods=1).mean()
        rolling_std = df[column].rolling(window=period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        logger.info(f"Calculated Bollinger Bands ({period}, {std_dev})")
        return df
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range (ATR)"""
        df = self.data.copy()
        
        # True Range calculations
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        # Maximum of the three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period, min_periods=1).mean()
        
        logger.info(f"Calculated ATR with period: {period}")
        return df
    
    def calculate_stochastic(self, period=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic Oscillator"""
        df = self.data.copy()
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=period, min_periods=1).min()
        highest_high = df['high'].rolling(window=period, min_periods=1).max()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        
        df['stoch_k_raw'] = 100 * ((df['close'] - lowest_low) / denominator)
        
        # Smooth %K
        df['stoch_k'] = df['stoch_k_raw'].rolling(window=smooth_k, min_periods=1).mean()
        
        # %D (smoothed %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_d, min_periods=1).mean()
        
        # Drop raw column
        df = df.drop(columns=['stoch_k_raw'])
        
        logger.info(f"Calculated Stochastic ({period}, {smooth_k}, {smooth_d})")
        return df
    
    def calculate_cci(self, period=20):
        """Calculate Commodity Channel Index (CCI)"""
        df = self.data.copy()
        
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Simple Moving Average of TP
        sma_tp = tp.rolling(window=period, min_periods=1).mean()
        
        # Mean Deviation
        def mean_deviation(x):
            return np.mean(np.abs(x - np.mean(x)))
        
        md = tp.rolling(window=period, min_periods=1).apply(mean_deviation)
        
        # CCI
        df['cci'] = (tp - sma_tp) / (0.015 * md)
        
        logger.info(f"Calculated CCI with period: {period}")
        return df
    
    def calculate_adx(self, period=14):
        """Calculate Average Directional Index (ADX)"""
        df = self.data.copy()
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period, min_periods=1).mean()
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        # +DM and -DM
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smooth DM
        df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period, min_periods=1).mean()
        df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period, min_periods=1).mean()
        
        # +DI and -DI (avoid division by zero)
        df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['atr'].replace(0, np.nan))
        df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['atr'].replace(0, np.nan))
        
        # DX
        di_sum = df['plus_di'] + df['minus_di']
        di_sum = di_sum.replace(0, np.nan)
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / di_sum
        
        # ADX
        df['adx'] = df['dx'].rolling(window=period, min_periods=1).mean()
        
        # Clean up intermediate columns
        cols_to_drop = ['up_move', 'down_move', 'plus_dm', 'minus_dm', 
                       'plus_dm_smooth', 'minus_dm_smooth', 'dx']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        logger.info(f"Calculated ADX with period: {period}")
        return df
    
    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV)"""
        df = self.data.copy()
        
        # Vectorized OBV calculation
        df['obv'] = 0
        price_direction = np.sign(df['close'].diff())
        df['obv'] = (price_direction * df['volume']).fillna(0).cumsum()
        
        df['obv_ma'] = df['obv'].rolling(window=20, min_periods=1).mean()
        
        logger.info("Calculated On-Balance Volume (OBV)")
        return df
    
    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index (MFI)"""
        df = self.data.copy()
        
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Raw Money Flow
        df['money_flow'] = tp * df['volume']
        
        # Positive and Negative Money Flow
        df['positive_flow'] = np.where(tp > tp.shift(1), df['money_flow'], 0)
        df['negative_flow'] = np.where(tp < tp.shift(1), df['money_flow'], 0)
        
        # Sum over period
        pos_sum = df['positive_flow'].rolling(window=period, min_periods=1).sum()
        neg_sum = df['negative_flow'].rolling(window=period, min_periods=1).sum()
        
        # Money Flow Ratio (avoid division by zero)
        neg_sum = neg_sum.replace(0, np.nan)
        money_ratio = pos_sum / neg_sum
        
        # MFI
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Clean up
        df = df.drop(columns=['money_flow', 'positive_flow', 'negative_flow'])
        
        logger.info(f"Calculated Money Flow Index (MFI) with period: {period}")
        return df
    
    def identify_support_resistance(self, window=20, num_levels=5):
        """Identify support and resistance levels"""
        df = self.data.copy()
        
        # Find local maxima and minima
        df['local_max'] = df['high'].rolling(window=window, center=True).max()
        df['local_min'] = df['low'].rolling(window=window, center=True).min()
        
        # Get unique levels
        resistance_levels = df[df['high'] == df['local_max']]['high'].dropna().unique()
        support_levels = df[df['low'] == df['local_min']]['low'].dropna().unique()
        
        # Sort and take top levels
        resistance = sorted(resistance_levels, reverse=True)[:num_levels]
        support = sorted(support_levels)[:num_levels]
        
        levels = {
            'resistance': [round(level, 2) for level in resistance],
            'support': [round(level, 2) for level in support]
        }
        
        logger.info(f"Identified support and resistance levels")
        return levels
    
    def detect_trend(self, short_period=20, long_period=50):
        """Detect trend direction"""
        df = self.calculate_sma(periods=[short_period, long_period])
        
        # Trend direction
        df['trend'] = np.where(
            df[f'sma_{short_period}'] > df[f'sma_{long_period}'], 1,
            np.where(df[f'sma_{short_period}'] < df[f'sma_{long_period}'], -1, 0)
        )
        
        # Trend strength (as percentage)
        df['trend_strength'] = abs(
            df[f'sma_{short_period}'] - df[f'sma_{long_period}']
        ) / df[f'sma_{long_period}'] * 100
        
        logger.info(f"Trend detection complete (SMA {short_period}/{long_period})")
        return df
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        logger.info("Calculating all technical indicators...")
        
        # Start with original data
        df = self.data.copy()
        self.data = df  # Update self.data to maintain consistency
        
        # Calculate each indicator (each method updates self.data)
        df = self.calculate_sma()
        self.data = df
        
        df = self.calculate_ema()
        self.data = df
        
        df = self.calculate_rsi()
        self.data = df
        
        df = self.calculate_macd()
        self.data = df
        
        df = self.calculate_stochastic()
        self.data = df
        
        df = self.calculate_cci()
        self.data = df
        
        df = self.calculate_mfi()
        self.data = df
        
        df = self.calculate_bollinger_bands()
        self.data = df
        
        df = self.calculate_atr()
        self.data = df
        
        df = self.calculate_obv()
        self.data = df
        
        df = self.calculate_adx()
        self.data = df
        
        df = self.detect_trend()
        self.data = df
        
        # Log all columns for debugging
        logger.info(f"All technical indicators calculated successfully! Total columns: {len(df.columns)}")
        logger.debug(f"Columns: {list(df.columns)}")
        
        return df
    
    def generate_signals(self):
        """Generate trading signals based on indicators"""
        # Calculate all indicators first
        df = self.calculate_all_indicators()
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0
        
        # Check if required columns exist before using them
        required_columns = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'macd_signal': 'MACD Signal',
            'bb_lower': 'Bollinger Lower',
            'bb_upper': 'Bollinger Upper',
            'stoch_k': 'Stochastic %K',
            'adx': 'ADX'
        }
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns for signal generation: {missing_columns}")
        
        # RSI signals
        if 'rsi' in df.columns:
            df.loc[df['rsi'] < self.params.get('rsi_oversold', 30), 'signal'] += 1
            df.loc[df['rsi'] > self.params.get('rsi_overbought', 70), 'signal'] -= 1
            logger.debug("Added RSI signals")
        
        # MACD signals
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            df.loc[macd_buy, 'signal'] += 1
            df.loc[macd_sell, 'signal'] -= 1
            logger.debug("Added MACD signals")
        
        # Bollinger Bands signals
        if all(col in df.columns for col in ['bb_lower', 'bb_upper', 'close']):
            df.loc[df['close'] < df['bb_lower'], 'signal'] += 1
            df.loc[df['close'] > df['bb_upper'], 'signal'] -= 1
            logger.debug("Added Bollinger Bands signals")
        
        # Stochastic signals
        if 'stoch_k' in df.columns:
            df.loc[df['stoch_k'] < 20, 'signal'] += 1
            df.loc[df['stoch_k'] > 80, 'signal'] -= 1
            logger.debug("Added Stochastic signals")
        
        # ADX trend strength
        if 'adx' in df.columns:
            df.loc[df['adx'] > 25, 'signal_strength'] += 1
            logger.debug("Added ADX trend strength")
        
        # Signal interpretation
        df['signal_action'] = 'HOLD'
        df.loc[df['signal'] >= 2, 'signal_action'] = 'STRONG BUY'
        df.loc[df['signal'] == 1, 'signal_action'] = 'BUY'
        df.loc[df['signal'] <= -2, 'signal_action'] = 'STRONG SELL'
        df.loc[df['signal'] == -1, 'signal_action'] = 'SELL'
        
        # Log signal distribution
        signal_counts = df['signal_action'].value_counts()
        logger.info(f"Trading signals generated - Distribution: {dict(signal_counts)}")
        
        return df


def analyze_stock(df, params=None):
    """Convenience function for technical analysis"""
    analyzer = TechnicalAnalyzer(df, params)
    return analyzer.calculate_all_indicators()


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Technical Analysis...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.randn(len(dates)) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Test the analyzer
    analyzer = TechnicalAnalyzer(sample_data)
    
    print("\n1. Testing calculate_all_indicators():")
    df_all = analyzer.calculate_all_indicators()
    print(f"   Columns: {len(df_all.columns)}")
    print(f"   Shape: {df_all.shape}")
    print(f"   Sample columns: {list(df_all.columns)[:10]}")
    
    print("\n2. Testing generate_signals():")
    df_signals = analyzer.generate_signals()
    print(f"   Signal distribution:")
    print(df_signals['signal_action'].value_counts())
    
    print("\nAll tests completed successfully!")
