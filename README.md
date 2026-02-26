# Stock Price Analysis Dashboard 📈

A comprehensive stock price analysis and predictive modeling system with real-time monitoring, technical indicators, risk metrics, and machine learning predictions.

## 🚀 Features

###  Data Collection
- Fetches historical data for 8 major stocks (AAPL, JPM, JNJ, MSFT, GOOGL, TSLA, AMZN, META)
- 5+ years of historical data
- Automatic caching and updates

###  Technical Analysis
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26, 50)
- **Momentum Indicators**: RSI, MACD, Stochastic, CCI
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, MFI
- **Trend Indicators**: ADX, Support/Resistance levels

###  Risk Metrics
- Returns (Total, Annualized, Daily)
- Volatility (Daily & Annual)
- Risk Ratios (Sharpe, Sortino)
- Maximum Drawdown
- Value at Risk (VaR) & CVaR
- Beta & Alpha (vs S&P 500)
- Correlation Matrix

###  Machine Learning
- **XGBoost** - Gradient boosting predictions
- **Random Forest** - Ensemble learning
- **Gradient Boosting** - Sequential learning
- **Ensemble Model** - Voting regressor combining all models
- Feature importance analysis
- 30-day price forecasts
<img width="1599" height="744" alt="image" src="https://github.com/user-attachments/assets/b89cd6aa-9716-41c7-beeb-a83400aa0b9a" />


###  Real-Time Monitoring
- Live price updates via WebSocket
- Automated alerts for:
  - Golden Cross (50 SMA crosses above 200 SMA)
  - Death Cross (50 SMA crosses below 200 SMA)
  - RSI Oversold/Overbought
  - Volume spikes
  - Price movements
<img width="1598" height="745" alt="image" src="https://github.com/user-attachments/assets/89d0dbf1-89ca-4a2e-a586-ca180d0998b6" />


### Portfolio Optimization
- Modern Portfolio Theory implementation
- Max Sharpe ratio optimization
- Efficient frontier visualization
- Weight distribution tables
- Investment allocation ($10k example)
<img width="1591" height="747" alt="image" src="https://github.com/user-attachments/assets/246e38c4-c1a0-43db-a65d-85634d819509" />


### Interactive Dashboard
- Built with Flask + Socket.IO
- Real-time updates
- Responsive design
- Dark theme
- Interactive Plotly charts
<img width="1600" height="742" alt="image" src="https://github.com/user-attachments/assets/771b5c49-dba9-4422-92ba-d687c9c64156" />


  


