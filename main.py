"""
Stock Price Analysis - Main Entry Point
Comprehensive analysis of historical stock data with technical indicators,
risk metrics, time series forecasting, and machine learning models
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCKS, MARKET_INDEX, START_DATE, END_DATE, DATA_DIR
from data_collection import StockDataCollector
from data_preparation import DataPreparator
from technical_analysis import TechnicalAnalyzer
from risk_metrics import RiskMetrics
from time_series_analysis import TimeSeriesAnalyzer
from ml_models import StockPricePredictor
from visualizations import StockVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalysis:
    """Main class for stock analysis workflow"""
    
    def __init__(self):
        """Initialize the analysis"""
        self.collector = StockDataCollector()
        self.stock_data = {}
        self.market_data = None
        self.prepared_data = {}
        self.technical_results = {}
        self.risk_results = {}
        self.ts_results = {}
        self.ml_results = {}
        
    def load_data(self, use_cache=True):
        """
        Load stock data
        
        Args:
            use_cache: Whether to use cached data
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Stock Data")
        logger.info("=" * 60)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if use_cache:
            # Try to load from CSV files
            for symbol in STOCKS.keys():
                filepath = os.path.join(DATA_DIR, f'{symbol}_data.csv')
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    self.stock_data[symbol] = df
                    logger.info(f"Loaded {symbol} data from cache")
            
            # Load market data
            market_path = os.path.join(DATA_DIR, 'market_data.csv')
            if os.path.exists(market_path):
                self.market_data = pd.read_csv(market_path, index_col=0, parse_dates=True)
                logger.info("Loaded market data from cache")
        
        if not self.stock_data:
            # Fetch fresh data
            logger.info("Fetching fresh data from API...")
            self.collector.fetch_all_stocks()
            self.stock_data = self.collector.stock_data
            self.market_data = self.collector.market_data
            self.collector.save_to_csv()
        
        logger.info(f"Loaded data for {len(self.stock_data)} stocks")
        
    def prepare_data(self):
        """
        Prepare and clean data
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Data Preparation")
        logger.info("=" * 60)
        
        preparator = DataPreparator()
        self.prepared_data = {}
        
        for symbol, df in self.stock_data.items():
            logger.info(f"Preparing {symbol} data...")
            prepared = preparator.prepare_for_analysis(df)
            self.prepared_data[symbol] = prepared
            
            # Calculate statistics
            stats = preparator.calculate_statistics(prepared)
            logger.info(f"\n{symbol} Statistics:")
            logger.info(f"\n{stats}")
        
    def run_technical_analysis(self):
        """
        Run technical analysis
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Technical Analysis")
        logger.info("=" * 60)
        
        self.technical_results = {}
        
        for symbol, df in self.prepared_data.items():
            logger.info(f"Analyzing {symbol}...")
            
            analyzer = TechnicalAnalyzer(df)
            df_tech = analyzer.calculate_all_indicators()
            
            # Generate signals
            df_signals = analyzer.generate_signals()
            
            self.technical_results[symbol] = {
                'data': df_tech,
                'signals': df_signals
            }
            
            logger.info(f"Technical indicators calculated for {symbol}")
    
    def calculate_risk_metrics(self):
        """
        Calculate risk and performance metrics
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Risk & Performance Metrics")
        logger.info("=" * 60)
        
        calculator = RiskMetrics(
            self.stock_data, 
            self.market_data
        )
        
        # Generate comprehensive report
        metrics_report = calculator.generate_metrics_report(self.stock_data)
        
        logger.info(f"\nRisk Metrics Report:")
        logger.info(f"\n{metrics_report}")
        
        # Save to file
        report_path = os.path.join(DATA_DIR, 'risk_metrics.csv')
        metrics_report.to_csv(report_path)
        logger.info(f"Risk metrics saved to {report_path}")
        
        # Correlation matrix
        corr_matrix = calculator.calculate_correlation_matrix(self.stock_data)
        logger.info(f"\nCorrelation Matrix:")
        logger.info(f"\n{corr_matrix}")
        
        # Save correlation matrix
        corr_path = os.path.join(DATA_DIR, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_path)
        
        self.risk_results = {
            'metrics': metrics_report,
            'correlation': corr_matrix
        }
    
    def run_time_series_analysis(self, symbol='AAPL'):
        """
        Run time series analysis and forecasting
        
        Args:
            symbol: Stock symbol to analyze
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Time Series Analysis")
        logger.info("=" * 60)
        
        if symbol not in self.prepared_data:
            logger.warning(f"Symbol {symbol} not found, using first available")
            symbol = list(self.prepared_data.keys())[0]
        
        df = self.prepared_data[symbol]
        
        # Initialize analyzer with close price column
        analyzer = TimeSeriesAnalyzer(df, column='close')
        
        # Stationarity test
        stationarity = analyzer.test_stationarity()
        logger.info(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
        logger.info(f"P-Value: {stationarity['p_value']:.4f}")
        logger.info(f"Is Stationary: {stationarity['is_stationary']}")
        
        # Find best ARIMA params
        try:
            # Try different method names based on your implementation
            if hasattr(analyzer, 'find_best_arima'):
                best_result = analyzer.find_best_arima()
                best_order = best_result.get('best_order', best_result)
            elif hasattr(analyzer, 'find_best_arima_params'):
                best_result = analyzer.find_best_arima_params()
                best_order = best_result.get('order', best_result)
            else:
                # Default to (1,1,1) if method not found
                best_order = (1, 1, 1)
                logger.warning("Could not find best ARIMA method, using default (1,1,1)")
            
            logger.info(f"Best ARIMA Order: {best_order}")
            
            # Fit model
            model = analyzer.fit_arima(best_order)
            
            # Forecast
            forecast = analyzer.forecast(steps=30)
            logger.info(f"\n30-Day Forecast (first 10 days):")
            logger.info(f"\n{forecast.head(10)}")
            
            self.ts_results = {
                'symbol': symbol,
                'stationarity': stationarity,
                'best_order': best_order,
                'forecast': forecast
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            self.ts_results = {
                'symbol': symbol,
                'stationarity': stationarity,
                'error': str(e)
            }
    
    def run_ml_models(self, symbol='AAPL'):
        """
        Run machine learning models
        
        Args:
            symbol: Stock symbol to predict
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Machine Learning Models")
        logger.info("=" * 60)
        
        if symbol not in self.prepared_data:
            symbol = list(self.prepared_data.keys())[0]
        
        df = self.prepared_data[symbol]
        
        try:
            predictor = StockPricePredictor(df)
            
            # Train all models
            logger.info("Training models...")
            results = predictor.train_all_models()
            
            # Compare models
            comparison = predictor.compare_models()
            logger.info(f"\nModel Comparison:")
            logger.info(f"\n{comparison}")
            
            # Backtest strategy
            backtest = predictor.backtest_strategy(initial_capital=10000)
            logger.info(f"\nBacktest Results:")
            logger.info(f"Strategy Return: {backtest['strategy_return']:.2%}")
            logger.info(f"Buy & Hold Return: {backtest['buy_hold_return']:.2%}")
            logger.info(f"Strategy Sharpe: {backtest['strategy_sharpe']:.4f}")
            
            self.ml_results[symbol] = {
                'comparison': comparison,
                'backtest': backtest
            }
            
        except Exception as e:
            logger.error(f"Error in ML models for {symbol}: {e}")
            self.ml_results[symbol] = {'error': str(e)}
    
    def create_visualizations(self, symbol='AAPL'):
        """
        Create visualizations
        
        Args:
            symbol: Stock symbol to visualize
        """
        logger.info("=" * 60)
        logger.info("STEP 7: Creating Visualizations")
        logger.info("=" * 60)
        
        if symbol not in self.stock_data:
            symbol = list(self.stock_data.keys())[0]
        
        df = self.stock_data[symbol]
        name = STOCKS.get(symbol, {}).get('name', symbol)
        
        visualizer = StockVisualizer(df, name)
        
        # Create basic plots
        logger.info(f"Creating visualizations for {symbol}...")
        
        try:
            # Try different visualization methods
            if hasattr(visualizer, 'plot_interactive_price'):
                fig = visualizer.plot_interactive_price()
            elif hasattr(visualizer, 'plot_price_history'):
                fig = visualizer.plot_price_history()
            else:
                logger.warning("No suitable visualization method found")
                fig = None
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizer
    
    def save_results(self):
        """Save all results to CSV files"""
        logger.info("=" * 60)
        logger.info("Saving Results")
        logger.info("=" * 60)
        
        # Save technical analysis results
        for symbol, results in self.technical_results.items():
            tech_path = os.path.join(DATA_DIR, f'{symbol}_technical.csv')
            results['data'].to_csv(tech_path)
            logger.info(f"Saved technical data for {symbol}")
        
        # Save time series results if available
        if hasattr(self, 'ts_results') and self.ts_results and 'forecast' in self.ts_results:
            forecast_path = os.path.join(DATA_DIR, f"{self.ts_results['symbol']}_forecast.csv")
            self.ts_results['forecast'].to_csv(forecast_path)
            logger.info(f"Saved forecast for {self.ts_results['symbol']}")
    
    def run_full_analysis(self, symbol='AAPL'):
        """
        Run complete analysis pipeline
        
        Args:
            symbol: Primary stock symbol to analyze
        """
        logger.info("=" * 60)
        logger.info("STOCK PRICE ANALYSIS - FULL PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Analyzing: {symbol}")
        logger.info(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
        
        # Run all steps
        self.load_data(use_cache=True)
        self.prepare_data()
        self.run_technical_analysis()
        self.calculate_risk_metrics()
        self.run_time_series_analysis(symbol)
        self.run_ml_models(symbol)
        self.create_visualizations(symbol)
        self.save_results()
        
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 60)


def main():
    """Main function"""
    # Create analysis instance
    analysis = StockAnalysis()
    
    # Run full analysis on AAPL (can be changed to JPM, JNJ)
    analysis.run_full_analysis(symbol='AAPL')
    
    # Also run for other stocks
    for symbol in ['JPM', 'JNJ']:
        try:
            logger.info(f"\n\n{'='*60}")
            logger.info(f"Additional Analysis: {symbol}")
            logger.info(f"{'='*60}\n")
            analysis.run_ml_models(symbol)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nResults saved in the 'data' directory.")


if __name__ == "__main__":
    main()
