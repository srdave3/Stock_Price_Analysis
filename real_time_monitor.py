"""
Real-Time Stock Monitoring Module
Alerts for technical indicators and price movements
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
import schedule
from datetime import datetime, timedelta
from threading import Thread
from plyer import notification  # For desktop notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import ALERT_THRESHOLDS, STOCKS

logger = logging.getLogger(__name__)


class Alert:
    """Alert class for notifications"""
    
    def __init__(self, symbol, alert_type, message, value, threshold):
        self.symbol = symbol
        self.alert_type = alert_type
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.symbol}: {self.message}"


class RealTimeMonitor:
    """Real-time stock monitoring with alerts"""
    
    def __init__(self, symbols=None, email_config=None):
        """
        Initialize the monitor
        
        Args:
            symbols: List of stock symbols to monitor
            email_config: Dictionary with email settings
        """
        self.symbols = symbols or list(STOCKS.keys())
        self.email_config = email_config
        self.thresholds = ALERT_THRESHOLDS
        
        # Store historical data
        self.historical_data = {}
        self.alerts = []
        self.alert_callbacks = []
        
        # Initialize historical data
        self._init_historical_data()
        
    def _init_historical_data(self):
        """Initialize historical data for each symbol"""
        for symbol in self.symbols:
            try:
                # Get 60 days of historical data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="60d", interval="1d")
                self.historical_data[symbol] = df
                logger.info(f"Initialized historical data for {symbol}")
            except Exception as e:
                logger.error(f"Error initializing {symbol}: {e}")
    
    def fetch_current_data(self, symbol):
        """
        Fetch current market data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with current data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get intraday data (1m interval for latest)
            current = ticker.history(period="1d", interval="1m")
            
            if not current.empty:
                # Get latest price
                latest = current.iloc[-1]
                
                # Calculate average volume
                hist = self.historical_data.get(symbol)
                avg_volume = hist['Volume'].mean() if hist is not None else latest['Volume']
                
                return {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'open': current.iloc[0]['Open'],
                    'high': current['High'].max(),
                    'low': current['Low'].min(),
                    'volume': latest['Volume'],
                    'avg_volume': avg_volume,
                    'change': (latest['Close'] - current.iloc[0]['Open']) / current.iloc[0]['Open'],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    def calculate_technical_indicators(self, symbol):
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with indicator values
        """
        df = self.historical_data.get(symbol)
        if df is None or len(df) < 20:
            return {}
        
        indicators = {}
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = exp1.iloc[-1] - exp2.iloc[-1]
        indicators['macd_signal'] = indicators['macd']
        
        # Moving averages
        indicators['sma_20'] = df['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = df['Close'].rolling(window=50).mean().iloc[-1]
        
        # Current price
        indicators['current_price'] = df['Close'].iloc[-1]
        
        # Volume spike
        avg_volume = df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        indicators['volume_ratio'] = current_volume / avg_volume
        
        return indicators
    
    def check_alerts(self, symbol, current_data, indicators):
        """
        Check for alert conditions
        
        Args:
            symbol: Stock symbol
            current_data: Current market data
            indicators: Technical indicators
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        if not current_data or not indicators:
            return alerts
        
        # Price change alert
        if abs(current_data['change']) * 100 >= self.thresholds.get('price_change_percent', 5):
            alerts.append(Alert(
                symbol,
                'PRICE_MOVE',
                f"Price moved {current_data['change']*100:.1f}%",
                current_data['change'],
                self.thresholds['price_change_pct']
            ))
        
        # RSI alerts
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi <= self.thresholds['rsi_oversold']:
                alerts.append(Alert(
                    symbol,
                    'RSI_OVERSOLD',
                    f"RSI oversold: {rsi:.1f}",
                    rsi,
                    self.thresholds['rsi_oversold']
                ))
            elif rsi >= self.thresholds['rsi_overbought']:
                alerts.append(Alert(
                    symbol,
                    'RSI_OVERBOUGHT',
                    f"RSI overbought: {rsi:.1f}",
                    rsi,
                    self.thresholds['rsi_overbought']
                ))
        
        # Volume spike alert
        if indicators.get('volume_ratio', 0) >= self.thresholds['volume_spike']:
            alerts.append(Alert(
                symbol,
                'VOLUME_SPIKE',
                f"Volume spike: {indicators['volume_ratio']:.1f}x average",
                indicators['volume_ratio'],
                self.thresholds['volume_spike']
            ))
        
        # SMA crossover
        if 'sma_20' in indicators and 'sma_50' in indicators:
            if indicators['sma_20'] > indicators['sma_50']:
                alerts.append(Alert(
                    symbol,
                    'GOLDEN_CROSS',
                    "Golden Cross (20 SMA > 50 SMA)",
                    indicators['sma_20'] / indicators['sma_50'] - 1,
                    0
                ))
            elif indicators['sma_20'] < indicators['sma_50']:
                alerts.append(Alert(
                    symbol,
                    'DEATH_CROSS',
                    "Death Cross (20 SMA < 50 SMA)",
                    indicators['sma_20'] / indicators['sma_50'] - 1,
                    0
                ))
        
        return alerts
    
    def send_notification(self, alert):
        """
        Send notification for alert
        
        Args:
            alert: Alert object
        """
        # Desktop notification
        try:
            notification.notify(
                title=f"Stock Alert: {alert.symbol}",
                message=alert.message,
                timeout=5
            )
        except:
            pass
        
        # Email notification (if configured)
        if self.email_config:
            self._send_email(alert)
        
        # Log alert
        logger.info(str(alert))
        
        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except:
                pass
    
    def _send_email(self, alert):
        """Send email notification"""
        if not self.email_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Stock Alert: {alert.symbol} - {alert.alert_type}"
            
            body = f"""
            <h2>Stock Alert Triggered</h2>
            <p><strong>Symbol:</strong> {alert.symbol}</p>
            <p><strong>Alert Type:</strong> {alert.alert_type}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            <p><strong>Value:</strong> {alert.value:.4f}</p>
            <p><strong>Threshold:</strong> {alert.threshold}</p>
            <p><strong>Time:</strong> {alert.timestamp}</p>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email error: {e}")
    
    def monitor_symbol(self, symbol):
        """
        Monitor a single symbol
        
        Args:
            symbol: Stock symbol
        """
        current_data = self.fetch_current_data(symbol)
        indicators = self.calculate_technical_indicators(symbol)
        
        alerts = self.check_alerts(symbol, current_data, indicators)
        
        for alert in alerts:
            self.alerts.append(alert)
            self.send_notification(alert)
        
        # Update historical data
        if current_data:
            # Simple update - in production, you'd want to properly update the DataFrame
            pass
    
    def monitor_all(self):
        """Monitor all symbols"""
        logger.info(f"Running monitor at {datetime.now()}")
        
        for symbol in self.symbols:
            try:
                self.monitor_symbol(symbol)
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
    
    def start(self, interval_minutes=5):
        """
        Start continuous monitoring
        
        Args:
            interval_minutes: Monitoring interval in minutes
        """
        logger.info(f"Starting real-time monitor (interval: {interval_minutes} minutes)")
        
        # Run immediately
        self.monitor_all()
        
        # Schedule regular runs
        schedule.every(interval_minutes).minutes.do(self.monitor_all)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def start_background(self, interval_minutes=5):
        """
        Start monitoring in background thread
        
        Args:
            interval_minutes: Monitoring interval
            
        Returns:
            Thread object
        """
        def run():
            self.start(interval_minutes)
        
        thread = Thread(target=run, daemon=True)
        thread.start()
        return thread
    
    def get_recent_alerts(self, n=10):
        """Get most recent alerts"""
        return sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def register_callback(self, callback):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)


# WebSocket server for real-time updates
class AlertWebSocket:
    """WebSocket server for streaming alerts"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.clients = []
        
    def start(self, host='localhost', port=8765):
        """Start WebSocket server"""
        import asyncio
        import websockets
        
        async def handler(websocket, path):
            self.clients.append(websocket)
            try:
                async for message in websocket:
                    # Handle client messages
                    pass
            finally:
                self.clients.remove(websocket)
        
        async def broadcast_alerts():
            while True:
                if self.clients:
                    alerts = self.monitor.get_recent_alerts(5)
                    alert_data = [
                        {
                            'symbol': a.symbol,
                            'type': a.alert_type,
                            'message': a.message,
                            'timestamp': a.timestamp.isoformat()
                        }
                        for a in alerts
                    ]
                    
                    # Broadcast to all clients
                    for client in self.clients:
                        try:
                            await client.send(json.dumps(alert_data))
                        except:
                            pass
                
                await asyncio.sleep(5)  # Update every 5 seconds
        
        async def main():
            async with websockets.serve(handler, host, port):
                await broadcast_alerts()
        
        thread = Thread(target=lambda: asyncio.run(main()), daemon=True)
        thread.start()
        logger.info(f"WebSocket server started on ws://{host}:{port}")


if __name__ == "__main__":
    # Test the monitor
    print("Testing Real-Time Monitor...")
    
    monitor = RealTimeMonitor(symbols=['AAPL', 'MSFT'])
    
    # Test single check
    print("\nChecking AAPL...")
    data = monitor.fetch_current_data('AAPL')
    indicators = monitor.calculate_technical_indicators('AAPL')
    
    if data:
        print(f"Price: ${data['price']:.2f}")
        print(f"Change: {data['change']*100:.2f}%")
    
    if indicators:
        print(f"RSI: {indicators.get('rsi', 'N/A'):.1f}")
        print(f"Volume Ratio: {indicators.get('volume_ratio', 'N/A'):.2f}")
    
    # Test alerts
    alerts = monitor.check_alerts('AAPL', data, indicators)
    for alert in alerts:
        print(f"Alert: {alert}")
    
    print("\nMonitor ready. To start continuous monitoring, use monitor.start()")