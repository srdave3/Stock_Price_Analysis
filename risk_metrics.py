"""
Risk and Performance Metrics Module for Stock Price Analysis
Calculates returns, volatility, Sharpe ratio, drawdown, beta, and correlations
"""

import pandas as pd
import numpy as np
import logging
from config import RISK_FREE_RATE

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Class for calculating risk and performance metrics"""
    
    def __init__(self, stock_data, market_data=None, risk_free_rate=None):
        """
        Initialize the risk metrics calculator
        
        Args:
            stock_data: DataFrame or dict of DataFrames with stock data
            market_data: DataFrame with market index data
            risk_free_rate: Annual risk-free rate
        """
        self.stock_data = stock_data
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate or RISK_FREE_RATE
        self.trading_days = 252
    
    def _ensure_datetime_index(self, df):
        """
        Ensure DataFrame has datetime index
        
        Args:
            df: DataFrame to check
            
        Returns:
            DataFrame with datetime index
        """
        if df is None:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.copy()
            df.index = pd.to_datetime(df.index,utc=True)
            df.index = df.index.tz_localize(None)
        return df
    
    def _prepare_dataframe(self, df):
        """
        Prepare DataFrame for calculations (ensure datetime index)
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Prepared DataFrame
        """
        return self._ensure_datetime_index(df)
    
    def calculate_returns(self, df, column='close'):
        """
        Calculate daily returns
        
        Args:
            df: DataFrame with price data
            column: Column to calculate returns for
            
        Returns:
            DataFrame with returns
        """
        df = self._prepare_dataframe(df)
        df_returns = df.copy()
        
        # Daily returns (percentage)
        df_returns['daily_return'] = df[column].pct_change()
        
        # Log returns
        df_returns['log_return'] = np.log(df[column] / df[column].shift(1))
        
        return df_returns
    
    def calculate_annualized_returns(self, df, column='close'):
        """
        Calculate annualized returns
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            
        Returns:
            Dictionary of return metrics
        """
        df = self._prepare_dataframe(df)
        
        # Daily returns
        daily_returns = df[column].pct_change().dropna()
        
        # Total return
        total_return = (df[column].iloc[-1] / df[column].iloc[0]) - 1
        
        # Annualized return
        try:
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                years = days / 365.25
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0
        except Exception as e:
            logger.warning(f"Error calculating annualized return: {e}")
            annualized_return = 0
            years = 1
        
        # Average daily return
        avg_daily_return = daily_returns.mean()
        
        # Annualized average return
        annualized_avg_return = avg_daily_return * self.trading_days
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_daily_return': avg_daily_return,
            'annualized_avg_return': annualized_avg_return
        }
    
    def calculate_volatility(self, df, column='close', annualize=True):
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            annualize: Whether to annualize the volatility
            
        Returns:
            Dictionary of volatility metrics
        """
        df = self._prepare_dataframe(df)
        daily_returns = df[column].pct_change().dropna()
        
        # Daily volatility
        daily_vol = daily_returns.std()
        
        # Annualized volatility
        if annualize:
            annual_vol = daily_vol * np.sqrt(self.trading_days)
        else:
            annual_vol = None
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol
        }
    
    def calculate_sharpe_ratio(self, df, column='close'):
        """
        Calculate Sharpe Ratio
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            
        Returns:
            Sharpe ratio
        """
        df = self._prepare_dataframe(df)
        returns = self.calculate_returns(df, column)
        daily_returns = returns['daily_return'].dropna()
        
        # Average excess return
        avg_excess_return = daily_returns.mean() - (self.risk_free_rate / self.trading_days)
        
        # Standard deviation of returns
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0
        
        # Sharpe ratio (annualized)
        sharpe_ratio = (avg_excess_return / std_return) * np.sqrt(self.trading_days)
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, df, column='close'):
        """
        Calculate Sortino Ratio (uses downside deviation)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            
        Returns:
            Sortino ratio
        """
        df = self._prepare_dataframe(df)
        returns = self.calculate_returns(df, column)
        daily_returns = returns['daily_return'].dropna()
        
        # Average excess return
        avg_excess_return = daily_returns.mean() - (self.risk_free_rate / self.trading_days)
        
        # Downside deviation (only negative returns)
        downside_returns = daily_returns[daily_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        downside_std = downside_returns.std()
        
        # Sortino ratio (annualized)
        sortino_ratio = (avg_excess_return / downside_std) * np.sqrt(self.trading_days)
        
        return sortino_ratio
    
    def calculate_max_drawdown(self, df, column='close'):
        """
        Calculate Maximum Drawdown
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            
        Returns:
            Dictionary with drawdown metrics
        """
        df = self._prepare_dataframe(df)
        
        # Calculate cumulative maximum
        rolling_max = df[column].cummax()
        
        # Calculate drawdown
        drawdown = (df[column] - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Peak (date and price)
        if not pd.isna(max_drawdown) and max_drawdown < 0:
            peak_idx = rolling_max[:df[column].idxmin()].idxmax()
            peak_price = rolling_max.loc[peak_idx]
            
            # Trough (date and price)
            trough_idx = df[column].idxmin()
            trough_price = df[column].min()
            
            # Recovery (if any)
            try:
                recovery_candidates = df.loc[trough_idx:][column] >= peak_price
                if recovery_candidates.any():
                    recovery_idx = recovery_candidates.idxmax()
                    recovery_return = (df.loc[recovery_idx, column] - trough_price) / trough_price
                else:
                    recovery_idx = None
                    recovery_return = None
            except:
                recovery_idx = None
                recovery_return = None
        else:
            peak_idx = None
            peak_price = None
            trough_idx = None
            trough_price = None
            recovery_idx = None
            recovery_return = None
        
        return {
            'max_drawdown': max_drawdown,
            'peak_date': peak_idx,
            'peak_price': peak_price,
            'trough_date': trough_idx,
            'trough_price': trough_price,
            'recovery_date': recovery_idx,
            'recovery_return': recovery_return,
            'drawdown_series': drawdown
        }
    
    def calculate_beta(self, stock_df, market_df, column='close'):
        """
        Calculate Beta (stock's volatility relative to market)
        
        Args:
            stock_df: DataFrame with stock data
            market_df: DataFrame with market data
            column: Column to calculate for
            
        Returns:
            Beta value
        """
        stock_df = self._prepare_dataframe(stock_df)
        market_df = self._prepare_dataframe(market_df)
        
        # Calculate returns
        stock_returns = stock_df[column].pct_change().dropna()
        market_returns = market_df[column].pct_change().dropna()
        
        # Align dates
        aligned_returns = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_returns) < 2:
            logger.warning("Insufficient data for beta calculation")
            return None
        
        # Calculate covariance and variance
        covariance = aligned_returns['stock'].cov(aligned_returns['market'])
        market_variance = aligned_returns['market'].var()
        
        if market_variance == 0:
            return None
        
        # Beta
        beta = covariance / market_variance
        
        return beta
    
    def calculate_alpha(self, stock_df, market_df, column='close'):
        """
        Calculate Alpha (excess return over expected return)
        
        Args:
            stock_df: DataFrame with stock data
            market_df: DataFrame with market data
            column: Column to calculate for
            
        Returns:
            Alpha value
        """
        stock_df = self._prepare_dataframe(stock_df)
        market_df = self._prepare_dataframe(market_df)
        
        # Calculate returns
        stock_annual_return = self.calculate_annualized_returns(stock_df, column)['annualized_return']
        market_annual_return = self.calculate_annualized_returns(market_df, column)['annualized_return']
        
        # Calculate beta
        beta = self.calculate_beta(stock_df, market_df, column)
        
        if beta is None:
            return None
        
        # Alpha
        alpha = stock_annual_return - (self.risk_free_rate + beta * (market_annual_return - self.risk_free_rate))
        
        return alpha
    
    def calculate_correlation_matrix(self, stock_data_dict, column='close'):
        """
        Calculate correlation matrix between stocks
        
        Args:
            stock_data_dict: Dictionary of DataFrames
            column: Column to calculate for
            
        Returns:
            Correlation matrix DataFrame
        """
        # Extract close prices
        close_prices = pd.DataFrame()
        
        for symbol, df in stock_data_dict.items():
            df = self._prepare_dataframe(df)
            close_prices[symbol] = df[column]
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        logger.info(f"Correlation matrix:\n{corr_matrix}")
        
        return corr_matrix
    
    def calculate_var(self, df, column='close', confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            confidence: Confidence level
            
        Returns:
            VaR value
        """
        df = self._prepare_dataframe(df)
        returns = df[column].pct_change().dropna()
        
        if len(returns) == 0:
            return {'historical_var': None, 'parametric_var': None, 'confidence_level': confidence}
        
        # Historical VaR
        var = returns.quantile(1 - confidence)
        
        # Parametric VaR (assuming normal distribution)
        z_score = 1.65 if confidence == 0.95 else 2.33
        parametric_var = returns.mean() - z_score * returns.std()
        
        return {
            'historical_var': var,
            'parametric_var': parametric_var,
            'confidence_level': confidence
        }
    
    def calculate_cvar(self, df, column='close', confidence=0.95):
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            df: DataFrame with price data
            column: Column to calculate for
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        df = self._prepare_dataframe(df)
        returns = df[column].pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        var = returns.quantile(1 - confidence)
        
        # CVaR (average of returns below VaR)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    def calculate_information_ratio(self, stock_df, market_df, column='close'):
        """
        Calculate Information Ratio
        
        Args:
            stock_df: DataFrame with stock data
            market_df: DataFrame with market data
            column: Column to calculate for
            
        Returns:
            Information ratio
        """
        stock_df = self._prepare_dataframe(stock_df)
        market_df = self._prepare_dataframe(market_df)
        
        # Calculate returns
        stock_returns = stock_df[column].pct_change().dropna()
        market_returns = market_df[column].pct_change().dropna()
        
        # Active returns (excess over benchmark)
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 2:
            return None
        
        active_returns = aligned['stock'] - aligned['market']
        
        if active_returns.std() == 0:
            return None
        
        # Information ratio (annualized)
        ir = (active_returns.mean() / active_returns.std()) * np.sqrt(self.trading_days)
        
        return ir
    
    def calculate_treynor_ratio(self, stock_df, market_df, column='close'):
        """
        Calculate Treynor Ratio
        
        Args:
            stock_df: DataFrame with stock data
            market_df: DataFrame with market data
            column: Column to calculate for
            
        Returns:
            Treynor ratio
        """
        stock_df = self._prepare_dataframe(stock_df)
        market_df = self._prepare_dataframe(market_df)
        
        # Calculate beta
        beta = self.calculate_beta(stock_df, market_df, column)
        
        if beta is None or beta == 0:
            return None
        
        # Calculate returns
        stock_annual_return = self.calculate_annualized_returns(stock_df, column)['annualized_return']
        
        # Treynor ratio
        treynor = (stock_annual_return - self.risk_free_rate) / beta
        
        return treynor
    
    def calculate_all_metrics(self, symbol, df):
        """
        Calculate all risk and performance metrics for a stock
        
        Args:
            symbol: Stock symbol
            df: DataFrame with stock data
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Calculating all metrics for {symbol}...")
        
        # Ensure datetime index
        df = self._prepare_dataframe(df)
        
        metrics = {'symbol': symbol}
        
        try:
            # Basic info
            metrics['start_date'] = df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None
            metrics['end_date'] = df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else None
            metrics['trading_days'] = len(df)
            
            # Calculate years for context
            if len(df) > 1:
                days = (df.index[-1] - df.index[0]).days
                metrics['years'] = round(days / 365.25, 2)
            else:
                metrics['years'] = 0
            
            # Returns
            returns = self.calculate_annualized_returns(df)
            for k, v in returns.items():
                if isinstance(v, float):
                    metrics[f'returns_{k}'] = round(v, 6)
                else:
                    metrics[f'returns_{k}'] = v
            
            # Volatility
            vol = self.calculate_volatility(df)
            for k, v in vol.items():
                if isinstance(v, float):
                    metrics[f'volatility_{k}'] = round(v, 6)
                else:
                    metrics[f'volatility_{k}'] = v
            
            # Risk metrics
            metrics['sharpe_ratio'] = round(self.calculate_sharpe_ratio(df), 4)
            metrics['sortino_ratio'] = round(self.calculate_sortino_ratio(df), 4)
            
            # Max Drawdown
            drawdown = self.calculate_max_drawdown(df)
            metrics['max_drawdown'] = round(drawdown['max_drawdown'], 6) if pd.notna(drawdown['max_drawdown']) else None
            
            # VaR
            var = self.calculate_var(df)
            metrics['var_95'] = round(var['historical_var'], 6) if var['historical_var'] is not None else None
            
            # CVaR
            cvar = self.calculate_cvar(df)
            metrics['cvar_95'] = round(cvar, 6) if cvar is not None else None
            
            # Market comparison
            if self.market_data is not None:
                market_df = self._prepare_dataframe(self.market_data)
                
                beta = self.calculate_beta(df, market_df)
                metrics['beta'] = round(beta, 4) if beta is not None else None
                
                alpha = self.calculate_alpha(df, market_df)
                metrics['alpha'] = round(alpha, 6) if alpha is not None else None
                
                ir = self.calculate_information_ratio(df, market_df)
                metrics['information_ratio'] = round(ir, 4) if ir is not None else None
                
                treynor = self.calculate_treynor_ratio(df, market_df)
                metrics['treynor_ratio'] = round(treynor, 4) if treynor is not None else None
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def generate_metrics_report(self, stock_data_dict):
        """
        Generate comprehensive metrics report for all stocks
        
        Args:
            stock_data_dict: Dictionary of stock DataFrames
            
        Returns:
            DataFrame with metrics for all stocks
        """
        logger.info("Generating metrics report...")
        
        report = {}
        
        for symbol, df in stock_data_dict.items():
            report[symbol] = self.calculate_all_metrics(symbol, df)
        
        report_df = pd.DataFrame(report).T
        
        logger.info(f"Metrics report generated with {len(report_df)} stocks")
        
        return report_df


def calculate_all_risks(stock_data, market_data=None):
    """
    Convenience function to calculate all risk metrics
    
    Args:
        stock_data: Stock DataFrame or dict of DataFrames
        market_data: Market DataFrame
        
    Returns:
        Metrics DataFrame or dictionary
    """
    calculator = RiskMetrics(stock_data, market_data)
    
    if isinstance(stock_data, dict):
        return calculator.generate_metrics_report(stock_data)
    else:
        symbol = stock_data.get('symbol', 'unknown') if hasattr(stock_data, 'get') else 'unknown'
        return calculator.calculate_all_metrics(symbol, stock_data)


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Risk Metrics...")
    
    # Create sample data with datetime index
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate random walk prices
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    sample_data = pd.DataFrame({
        'close': prices,
        'open': prices + np.random.randn(len(dates)),
        'high': prices + np.abs(np.random.randn(len(dates))),
        'low': prices - np.abs(np.random.randn(len(dates))),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Market data
    market_prices = 4000 + np.cumsum(np.random.randn(len(dates)) * 10)
    market_data = pd.DataFrame({
        'close': market_prices
    }, index=dates)
    
    # Test with string index (to simulate the issue)
    sample_data_str_index = sample_data.copy()
    sample_data_str_index.index = sample_data_str_index.index.strftime('%Y-%m-%d')
    
    print("\nTesting with string index (simulating your error case)...")
    calculator = RiskMetrics(sample_data_str_index, market_data)
    
    print("\n1. Annualized Returns:")
    ret = calculator.calculate_annualized_returns(sample_data_str_index)
    for k, v in ret.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    print("\n2. All Metrics (should work now with string index):")
    all_metrics = calculator.calculate_all_metrics('TEST', sample_data_str_index)
    for k, v in all_metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    print("\nTest completed successfully!")