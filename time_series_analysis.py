import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """Class for time series analysis of stock data"""
    
    def __init__(self, data, column='close'):
        """
        Initialize the time series analyzer
        
        Args:
            data: DataFrame with stock data
            column: Column to analyze
        """
        self.data = data.copy()
        self.column = column
        self.series = self.data[column]
        self.model_fit = None
        self.best_order = None
        
    def test_stationarity(self, series=None):
        """
        Test for stationarity using ADF test
        
        Args:
            series: Time series data (uses self.series if None)
            
        Returns:
            Dictionary with test results
        """
        if series is None:
            series = self.series
            
        # Drop NaN values
        series = series.dropna()
        
        # Perform ADF test
        result = adfuller(series, autolag='AIC')
        
        results = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        return results
    
    def analyze_acf_pacf(self, lags=40):
        """
        Analyze ACF and PACF
        
        Args:
            lags: Number of lags
            
        Returns:
            Dictionary with ACF and PACF values
        """
        series = self.series.dropna()
        
        # Calculate ACF and PACF
        acf_values = acf(series, nlags=lags, fft=True)
        pacf_values = pacf(series, nlags=lags, method='ywm')
        
        # Find significant lags (outside confidence interval)
        confidence_interval = 1.96 / np.sqrt(len(series))
        
        significant_acf = [i for i, val in enumerate(acf_values[1:], 1) 
                          if abs(val) > confidence_interval]
        significant_pacf = [i for i, val in enumerate(pacf_values[1:], 1) 
                           if abs(val) > confidence_interval]
        
        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'significant_acf': significant_acf[:10],  # Top 10
            'significant_pacf': significant_pacf[:10],  # Top 10
            'confidence_interval': confidence_interval
        }
    
    def find_best_arima(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        """
        Find best ARIMA parameters using AIC
        
        Args:
            p_range: Range of AR orders
            d_range: Range of differencing orders
            q_range: Range of MA orders
            
        Returns:
            Best order and corresponding AIC
        """
        series = self.series.dropna()
        
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # Try different orders
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        # Fit ARIMA model
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()
                        
                        # Check if this is the best model
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                            
                    except:
                        continue
        
        # Store the best model
        self.best_order = best_order
        self.model_fit = best_model
        
        return {
            'best_order': best_order,
            'best_aic': best_aic,
            'model_fit': best_model
        }
    
    def fit_arima(self, order):
        """
        Fit ARIMA model with specified order
        
        Args:
            order: Tuple (p, d, q)
            
        Returns:
            Fitted model
        """
        series = self.series.dropna()
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        self.model_fit = model.fit()
        self.best_order = order
        
        return self.model_fit
    
    def forecast(self, steps=30):
        """
        Generate forecasts using the fitted ARIMA model
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting. Run find_best_arima() or fit_arima() first.")
        
        # Get forecast
        forecast_result = self.model_fit.get_forecast(steps=steps)
        
        # Get forecast values and confidence intervals
        forecast_values = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        
        # Create forecast index
        last_date = self.data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast': forecast_values.values,
            'lower_ci': confidence_intervals.iloc[:, 0].values,
            'upper_ci': confidence_intervals.iloc[:, 1].values
        }, index=forecast_index)
        
        return forecast_df
    
    def decompose_series(self, model='additive', period=None):
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            model: 'additive' or 'multiplicative'
            period: Seasonal period (e.g., 5 for weekly, 252 for yearly)
            
        Returns:
            Decomposition result
        """
        if period is None:
            # Auto-detect period based on data frequency
            if len(self.series) > 252 * 2:  # More than 2 years
                period = 252  # Yearly for daily data
            elif len(self.series) > 90:  # More than 3 months
                period = 5  # Weekly
            else:
                period = 5  # Default to weekly
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            self.series.dropna(), 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        return decomposition
    
    def plot_forecast(self, steps=30, historical_days=100):
        """
        Plot historical data and forecast
        
        Args:
            steps: Number of steps to forecast
            historical_days: Number of historical days to show
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")
        
        # Get forecast
        forecast_df = self.forecast(steps)
        
        # Get historical data
        historical = self.series.tail(historical_days)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical.index, historical.values, 
                label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['forecast'], 
                label='Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_df.index, 
                         forecast_df['lower_ci'], 
                         forecast_df['upper_ci'], 
                         color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(f'Stock Price Forecast (ARIMA{self.best_order})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return forecast_df


# Test the analyzer
if __name__ == "__main__":
    print("Testing Time Series Analysis...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # Generate random walk with trend
    trend = np.linspace(0, 20, len(dates))
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Yearly cycle
    noise = np.random.randn(len(dates)) * 2
    
    prices = 100 + trend + seasonal + np.cumsum(noise) * 0.1
    
    sample_data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(sample_data)
    
    # 1. Stationarity Test
    print("\n1. Stationarity Test:")
    stationarity = analyzer.test_stationarity()
    print(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"P-Value: {stationarity['p_value']:.4f}")
    print(f"Is Stationary: {stationarity['is_stationary']}")
    
    # 2. ACF/PACF Analysis
    print("\n2. Autocorrelation Analysis:")
    acf_pacf = analyzer.analyze_acf_pacf(lags=20)
    print(f"Significant ACF lags: {acf_pacf['significant_acf']}")
    print(f"Significant PACF lags: {acf_pacf['significant_pacf']}")
    
    # 3. Find Best ARIMA Params
    print("\n3. Finding Best ARIMA Params:")
    best = analyzer.find_best_arima(p_range=(0, 2), d_range=(0, 1), q_range=(0, 2))
    print(f"Best Order: {best['best_order']}")
    print(f"Best AIC: {best['best_aic']:.2f}")
    
    # 4. Model Summary
    print("\n4. Fitted Model Summary:")
    print(analyzer.model_fit.summary())
    
    # 5. Forecast
    print("\n5. Forecasting:")
    forecast_df = analyzer.forecast(steps=30)
    print(forecast_df.head())
    
    # 6. Plot (optional - will display if you have matplotlib)
    try:
        analyzer.plot_forecast(steps=30, historical_days=100)
    except:
        print("\n(Plotting skipped - matplotlib issue)")
