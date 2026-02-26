"""
Data Collection Module for Stock Price Analysis
Fetches historical stock data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from config import STOCKS, MARKET_INDEX, START_DATE, END_DATE, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Class to collect and manage stock data"""
    
    def __init__(self, start_date=None, end_date=None):
        """
        Initialize the data collector
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.stock_data = {}
        self.market_data = None
        
    def fetch_stock_data(self, symbol, adjust_prices=True, period=None, start_date=None, end_date=None):
        """
        Fetch historical stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            adjust_prices: Whether to adjust for splits and dividends
            period: Optional period string (e.g., '1y', '6mo', '1mo', '5d')
            start_date: Start date (if period not used)
            end_date: End date (if period not used)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Determine which parameters to use
            if period:
                # Use period parameter
                logger.info(f"Using period='{period}' for {symbol}")
                df = ticker.history(
                    period=period,
                    auto_adjust=adjust_prices
                )
            else:
                # Use date range
                start = start_date or self.start_date
                end = end_date or self.end_date
                
                logger.info(f"Using date range {start} to {end} for {symbol}")
                df = ticker.history(
                    start=start,
                    end=end,
                    auto_adjust=adjust_prices
                )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column for identification
            df['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_market_data(self, symbol=None, period=None):
        """
        Fetch market index data for benchmarking
        
        Args:
            symbol: Market index symbol (default: S&P 500)
            period: Optional period string
            
        Returns:
            DataFrame with market index data
        """
        symbol = symbol or MARKET_INDEX
        logger.info(f"Fetching market data for {symbol}...")
        
        df = self.fetch_stock_data(symbol, period=period)
        
        if df is not None:
            self.market_data = df
            logger.info(f"Market data fetched successfully")
        
        return df
    
    def fetch_all_stocks(self, period=None):
        """
        Fetch data for all configured stocks
        
        Args:
            period: Optional period string to use instead of date range
            
        Returns:
            Dictionary of DataFrames
        """
        logger.info("Fetching data for all stocks...")
        
        for symbol in STOCKS.keys():
            if period:
                df = self.fetch_stock_data(symbol, period=period)
            else:
                df = self.fetch_stock_data(symbol)
            
            if df is not None:
                self.stock_data[symbol] = df
        
        # Also fetch market data
        self.fetch_market_data(period=period)
        
        logger.info(f"Successfully fetched data for {len(self.stock_data)} stocks")
        return self.stock_data
    
    def get_combined_data(self):
        """
        Get combined DataFrame of all stocks
        
        Returns:
            Combined DataFrame with all stock data
        """
        if not self.stock_data:
            self.fetch_all_stocks()
        
        dfs = []
        for symbol, df in self.stock_data.items():
            dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=False)
            return combined
        
        return None
    
    def save_to_csv(self, symbol=None, filepath=None):
        """
        Save stock data to CSV file
        
        Args:
            symbol: Stock symbol to save (None for all)
            filepath: Custom filepath
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if symbol:
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
            elif symbol == 'market':
                df = self.market_data
            else:
                logger.error(f"Unknown symbol: {symbol}")
                return saved_files
            
            if filepath is None:
                filename = f"{symbol}_data.csv"
                filepath = os.path.join(DATA_DIR, filename)
            
            df.to_csv(filepath)
            saved_files.append(filepath)
            logger.info(f"Saved {symbol} data to {filepath}")
        else:
            # Save all stocks
            for sym, df in self.stock_data.items():
                filename = f"{sym}_data.csv"
                filepath = os.path.join(DATA_DIR, filename)
                df.to_csv(filepath)
                saved_files.append(filepath)
            
            # Save market data
            if self.market_data is not None:
                filename = "market_data.csv"
                filepath = os.path.join(DATA_DIR, filename)
                self.market_data.to_csv(filepath)
                saved_files.append(filepath)
        
        return saved_files
    
    def load_from_csv(self, symbol):
        """
        Load stock data from CSV file
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with stock data
        """
        filepath = os.path.join(DATA_DIR, f"{symbol}_data.csv")
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.stock_data[symbol] = df
            logger.info(f"Loaded {symbol} data from {filepath}")
            return df
        else:
            logger.error(f"File not found: {filepath}")
            return None
    
    def get_latest_price(self, symbol):
        """
        Get the latest price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest closing price
        """
        if symbol in self.stock_data:
            return self.stock_data[symbol]['close'].iloc[-1]
        elif symbol == MARKET_INDEX and self.market_data is not None:
            return self.market_data['close'].iloc[-1]
        else:
            df = self.fetch_stock_data(symbol, period='5d')
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
        return None
    
    def get_ticker(self, symbol):
        """
        Get yfinance Ticker object for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            yfinance Ticker object
        """
        return yf.Ticker(symbol)
    
    def get_company_info(self, symbol):
        """
        Get company information for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            ticker = self.get_ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return company_info
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return None


def collect_stock_data(period=None):
    """
    Main function to collect all stock data
    
    Args:
        period: Optional period string
        
    Returns:
        StockDataCollector instance with loaded data
    """
    collector = StockDataCollector()
    
    if period:
        collector.fetch_all_stocks(period=period)
    else:
        collector.fetch_all_stocks()
    
    collector.save_to_csv()
    
    return collector


def quick_update(symbols=None, period='1mo'):
    """
    Quick update for recent data
    
    Args:
        symbols: List of symbols to update
        period: Period to fetch
        
    Returns:
        Dictionary of DataFrames
    """
    if symbols is None:
        symbols = list(STOCKS.keys())
    
    collector = StockDataCollector()
    data = {}
    
    for symbol in symbols:
        df = collector.fetch_stock_data(symbol, period=period)
        if df is not None:
            data[symbol] = df
    
    return data


if __name__ == "__main__":
    # Test the data collector
    print("Testing Stock Data Collector...")
    collector = StockDataCollector()
    
    # Test with period parameter
    print("\n1. Testing with period='1mo'...")
    aapl_data_period = collector.fetch_stock_data('AAPL', period='1mo')
    if aapl_data_period is not None:
        print(f"   Shape: {aapl_data_period.shape}")
        print(f"   Latest price: ${aapl_data_period['close'].iloc[-1]:.2f}")
    
    # Test with date range
    print("\n2. Testing with default date range...")
    aapl_data_range = collector.fetch_stock_data('AAPL')
    if aapl_data_range is not None:
        print(f"   Shape: {aapl_data_range.shape}")
        print(f"   Date range: {aapl_data_range.index[0]} to {aapl_data_range.index[-1]}")
    
    # Test get_ticker method
    print("\n3. Testing get_ticker method...")
    ticker = collector.get_ticker('AAPL')
    print(f"   Ticker object created: {ticker}")
    
    # Test company info
    print("\n4. Testing company info...")
    info = collector.get_company_info('AAPL')
    if info:
        print(f"   Company: {info['name']}")
        print(f"   Sector: {info['sector']}")
    
    # Fetch all stocks with period
    print("\n5. Testing fetch_all_stocks with period='6mo'...")
    collector.fetch_all_stocks(period='6mo')
    print(f"   Total stocks fetched: {len(collector.stock_data)}")
    
    print("\n✓ Data collection tests completed!")
