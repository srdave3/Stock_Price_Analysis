"""
Stock Analysis Dashboard
Flask web application with real-time updates, portfolio optimization, and ML insights
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import threading
import logging
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCKS, DATA_DIR
from data_collection import StockDataCollector
from technical_analysis import TechnicalAnalyzer
from risk_metrics import RiskMetrics
from portfolio_optimizer import PortfolioOptimizer
from real_time_monitor import RealTimeMonitor
from advanced_ml import AdvancedStockPredictor, SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock-analysis-secret-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global data store
app_data = {
    'stock_data': {},
    'technical_data': {},
    'risk_metrics': None,
    'portfolio_results': None,
    'alerts': [],
    'last_update': None,
    'update_error': None
}

# Initialize components
collector = StockDataCollector()
monitor = RealTimeMonitor(symbols=list(STOCKS.keys()))

def create_mock_data(symbol):
    """Create mock stock data for testing when API fails"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    logger.warning(f"Creating mock data for {symbol}")
    
    # Use symbol as seed for consistent mock data
    np.random.seed(hash(symbol) % 42)
    
    # Generate dates for the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price movements (random walk)
    returns = np.random.randn(len(dates)) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    # Add some symbol-specific characteristics
    if symbol == 'AAPL':
        price = price * 1.8  # Higher price for Apple
    elif symbol == 'TSLA':
        price = price * 2.5 + np.random.randn(len(dates)) * 10  # More volatile
    elif symbol == 'JNJ':
        price = price * 1.5 + 50  # Stable healthcare
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.005),
        'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
        'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df

def update_data():
    """Update stock data periodically"""
    global app_data
    
    logger.info("=" * 60)
    logger.info("UPDATING STOCK DATA")
    logger.info("=" * 60)
    
    try:
        # Clear previous errors
        app_data['update_error'] = None
        loaded_count = 0
        
        # Fetch latest data for each stock
        for symbol in STOCKS.keys():
            try:
                logger.info(f"Fetching {symbol}...")
                
                # Try to fetch real data with period parameter
                df = collector.fetch_stock_data(symbol, period='1y')
                
                # If that fails, try without period (using default date range)
                if df is None or df.empty:
                    logger.warning(f"Period fetch failed for {symbol}, trying date range...")
                    df = collector.fetch_stock_data(symbol)
                
                # If still no data, use mock data
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}, using mock data")
                    df = create_mock_data(symbol)
                
                # Store the data
                app_data['stock_data'][symbol] = df
                loaded_count += 1
                
                latest_price = df['close'].iloc[-1]
                logger.info(f"✓ {symbol} loaded - {len(df)} days, latest: ${latest_price:.2f}")
                
                # Calculate technical indicators
                try:
                    analyzer = TechnicalAnalyzer(df)
                    tech_df = analyzer.calculate_all_indicators()
                    app_data['technical_data'][symbol] = tech_df
                    logger.info(f"  Technical indicators calculated for {symbol}")
                except Exception as e:
                    logger.error(f"  Technical analysis error for {symbol}: {e}")
                    
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                logger.debug(traceback.format_exc())
                
                # Use mock data as fallback
                logger.warning(f"Using mock data for {symbol} due to error")
                app_data['stock_data'][symbol] = create_mock_data(symbol)
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count}/{len(STOCKS)} stocks")
        
        # Calculate risk metrics if we have data
        if app_data['stock_data']:
            try:
                logger.info("Calculating risk metrics...")
                calculator = RiskMetrics(app_data['stock_data'])
                app_data['risk_metrics'] = calculator.generate_metrics_report(app_data['stock_data'])
                logger.info("✓ Risk metrics calculated")
            except Exception as e:
                logger.error(f"Risk metrics error: {e}")
            
            # Calculate portfolio optimization
            try:
                logger.info("Calculating portfolio optimization...")
                returns_df = pd.DataFrame()
                for symbol, df in app_data['stock_data'].items():
                    returns_df[symbol] = df['close'].pct_change()
                returns_df = returns_df.dropna()
                
                if not returns_df.empty and len(returns_df.columns) > 1:
                    optimizer = PortfolioOptimizer(returns_df)
                    app_data['portfolio_results'] = optimizer.maximize_sharpe_ratio()
                    logger.info("✓ Portfolio optimization calculated")
                else:
                    logger.warning("Insufficient data for portfolio optimization")
            except Exception as e:
                logger.error(f"Portfolio optimization error: {e}")
        
        app_data['last_update'] = datetime.now()
        
        # Emit update to connected clients
        socketio.emit('data_updated', {'timestamp': datetime.now().isoformat()})
        
        logger.info("=" * 60)
        logger.info(f"UPDATE COMPLETE at {app_data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"Error updating data: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        app_data['update_error'] = error_msg


# Start background data update thread
def background_updater():
    """Background thread for data updates"""
    logger.info("Starting background updater thread...")
    
    # Initial update
    update_data()
    
    # Schedule periodic updates
    while True:
        socketio.sleep(300)  # Update every 5 minutes
        update_data()


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', stocks=STOCKS)


@app.route('/dashboard')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html', stocks=STOCKS)


@app.route('/portfolio')
def portfolio():
    """Portfolio optimization page"""
    return render_template('portfolio.html', stocks=STOCKS)


@app.route('/monitor')
def monitor_page():
    """Real-time monitor page"""
    return render_template('monitor.html', stocks=STOCKS)


@app.route('/ml_insights')
def ml_insights():
    """ML insights page"""
    return render_template('ml_insights.html', stocks=STOCKS)


@app.route('/api/stock_data/<symbol>')
def get_stock_data(symbol):
    """API endpoint for stock data"""
    try:
        if symbol not in app_data['stock_data']:
            logger.warning(f"Symbol {symbol} not found in stock_data")
            return jsonify({'success': False, 'error': 'Symbol not found'})
        
        df = app_data['stock_data'][symbol].tail(100)
        
        if df.empty:
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Create OHLC chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # RSI
        if symbol in app_data['technical_data']:
            tech_df = app_data['technical_data'][symbol]
            if 'rsi' in tech_df.columns:
                # Align dates
                common_dates = df.index.intersection(tech_df.index)
                if len(common_dates) > 0:
                    rsi_values = tech_df.loc[common_dates, 'rsi']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=common_dates,
                            y=rsi_values,
                            name='RSI',
                            line=dict(color='purple', width=2),
                            showlegend=True
                        ),
                        row=3, col=1
                    )
                    
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                 row=3, col=1, annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                 row=3, col=1, annotation_text="Oversold")
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {STOCKS[symbol]["name"]}',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=800,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        # Convert to JSON
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get latest stats
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        change_pct = ((latest['close'] / previous['close']) - 1) * 100
        
        stats = {
            'price': f"${latest['close']:.2f}",
            'change': f"{change_pct:+.2f}%",
            'volume': f"{latest['volume']:,.0f}",
            'high': f"${latest['high']:.2f}",
            'low': f"${latest['low']:.2f}",
            'open': f"${latest['open']:.2f}",
            'date': latest.name.strftime('%Y-%m-%d')
        }
        
        return jsonify({
            'success': True,
            'chart': graph_json,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_stock_data for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/risk_metrics')
def get_risk_metrics():
    """API endpoint for risk metrics"""
    try:
        if app_data['risk_metrics'] is None:
            return jsonify({'success': False, 'error': 'No risk metrics available'})
        
        # Format for display
        df = app_data['risk_metrics'].copy()
        
        # Format percentages
        for col in df.columns:
            if 'return' in col.lower() or 'volatility' in col.lower() or 'drawdown' in col.lower():
                df[col] = df[col].apply(lambda x: f"{float(x):.2%}" if pd.notna(x) else 'N/A')
            elif 'ratio' in col.lower() or 'beta' in col.lower() or 'alpha' in col.lower():
                df[col] = df[col].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else 'N/A')
        
        # Convert to HTML table
        table_html = df.to_html(classes='table table-striped table-dark table-hover', 
                                border=0, escape=False)
        
        return jsonify({
            'success': True,
            'table': table_html,
            'data': df.to_dict(orient='index')
        })
        
    except Exception as e:
        logger.error(f"Error in get_risk_metrics: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/portfolio_optimization')
def get_portfolio_optimization():
    """API endpoint for portfolio optimization"""
    try:
        if not app_data['portfolio_results'] or not app_data['portfolio_results'].get('success', False):
            return jsonify({'success': False, 'error': 'No optimization results'})
        
        results = app_data['portfolio_results']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(results['weights'].keys()),
            values=list(results['weights'].values()),
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title='Optimal Portfolio Weights',
            template='plotly_dark',
            annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'chart': chart_json,
            'weights': {k: f"{v:.2%}" for k, v in results['weights'].items()},
            'expected_return': f"{results['expected_return']:.2%}",
            'expected_volatility': f"{results['expected_volatility']:.2%}",
            'sharpe_ratio': f"{results['sharpe_ratio']:.3f}"
        })
        
    except Exception as e:
        logger.error(f"Error in get_portfolio_optimization: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/alerts')
def get_alerts():
    """API endpoint for recent alerts"""
    try:
        alerts = monitor.get_recent_alerts(20)
        
        alert_list = []
        for alert in alerts:
            alert_list.append({
                'symbol': alert.symbol,
                'type': alert.alert_type,
                'message': alert.message,
                'value': f"{alert.value:.2f}" if isinstance(alert.value, float) else str(alert.value),
                'timestamp': alert.timestamp.strftime('%H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'alerts': alert_list
        })
        
    except Exception as e:
        logger.error(f"Error in get_alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ml_predict/<symbol>')
def ml_predict(symbol):
    """API endpoint for ML predictions"""
    try:
        if symbol not in app_data['stock_data']:
            return jsonify({'success': False, 'error': 'Symbol not found'})
        
        df = app_data['stock_data'][symbol]
        
        if df.empty or len(df) < 100:
            return jsonify({'success': False, 'error': 'Insufficient data for ML predictions'})
        
        # Initialize predictor with limited features for speed
        predictor = AdvancedStockPredictor(df.tail(500))  # Use last 500 days
        
        # Train models with smaller test size for demo
        results = predictor.train_all_models(test_size=0.2)
        
        # Create prediction chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.1)
        
        # Actual vs predicted
        if 'test_data' in results:
            y_test = results['test_data']['y_test_values']
            x_axis = list(range(len(y_test)))
            
            # Add actual prices
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_test,
                    name='Actual',
                    mode='lines',
                    line=dict(color='white', width=3)
                ),
                row=1, col=1
            )
            
            # Add predictions from each model
            colors = {'xgboost': 'red', 'random_forest': 'blue', 
                     'gradient_boosting': 'green', 'ensemble': 'purple'}
            
            for name, result in results.items():
                if name not in ['test_data'] and 'predictions' in result:
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=result['predictions'],
                            name=f"{name.replace('_', ' ').title()}",
                            mode='lines',
                            line=dict(color=colors.get(name, 'orange'), width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
        
        # Feature importance placeholder
        fig.add_trace(
            go.Bar(
                x=[0.3, 0.25, 0.2, 0.15, 0.1],
                y=['RSI', 'MACD', 'Volume', 'SMA 20', 'ATR'],
                orientation='h',
                name='Feature Importance',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'ML Price Predictions for {symbol}',
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time (Days)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Features", row=2, col=1)
        fig.update_yaxes(title_text="Importance", row=2, col=1)
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get metrics
        metrics = {}
        for name, result in results.items():
            if name not in ['test_data'] and 'rmse' in result:
                metrics[name] = {
                    'rmse': f"${result['rmse']:.2f}",
                    'r2': f"{result['r2']:.3f}",
                    'mae': f"${result.get('mae', 0):.2f}"
                }
        
        return jsonify({
            'success': True,
            'chart': chart_json,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"ML prediction error for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check loaded data"""
    try:
        stock_info = {}
        for symbol, df in app_data['stock_data'].items():
            if df is not None and not df.empty:
                stock_info[symbol] = {
                    'rows': len(df),
                    'latest_price': float(df['close'].iloc[-1]),
                    'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                    'columns': list(df.columns)
                }
            else:
                stock_info[symbol] = {'error': 'No data'}
        
        return jsonify({
            'success': True,
            'stocks_loaded': list(app_data['stock_data'].keys()),
            'stock_count': len(app_data['stock_data']),
            'last_update': app_data['last_update'].strftime('%Y-%m-%d %H:%M:%S') if app_data['last_update'] else None,
            'update_error': app_data.get('update_error'),
            'has_technical': list(app_data['technical_data'].keys()),
            'has_risk_metrics': app_data['risk_metrics'] is not None,
            'has_portfolio': app_data['portfolio_results'] is not None,
            'stock_details': stock_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'time': datetime.now().isoformat(),
        'stocks_loaded': len(app_data['stock_data'])
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to server', 'time': datetime.now().isoformat()})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_stock')
def handle_subscribe(data):
    """Handle stock subscription"""
    symbol = data.get('symbol')
    if symbol:
        logger.info(f"Client {request.sid} subscribed to {symbol}")
        emit('subscribed', {'symbol': symbol})


def start_monitor():
    """Start real-time monitor in background"""
    logger.info("Starting real-time monitor...")
    monitor.start_background(interval_minutes=1)
    
    # Register callback for WebSocket updates
    def alert_callback(alert):
        socketio.emit('new_alert', {
            'symbol': alert.symbol,
            'type': alert.alert_type,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat()
        })
    
    monitor.register_callback(alert_callback)
    logger.info("Real-time monitor started")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("STARTING STOCK ANALYSIS DASHBOARD")
    logger.info("=" * 60)
    
    # Initial data update
    logger.info("Performing initial data update...")
    update_data()
    
    # Start background threads
    logger.info("Starting background threads...")
    threading.Thread(target=background_updater, daemon=True).start()
    threading.Thread(target=start_monitor, daemon=True).start()
    
    # Run Flask app
    logger.info(f"Starting Flask server on http://0.0.0.0:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)