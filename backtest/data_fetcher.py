"""
Data fetcher for cryptocurrency pairs
Supports multiple data sources for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import yfinance as yf
import requests
import time

class CryptoDataFetcher:
    """
    Fetch cryptocurrency data from various sources
    """
    
    def __init__(self, source='binance'):
        """
        Initialize data fetcher
        
        Args:
            source: Data source ('binance', 'yahoo', 'coingecko')
        """
        self.source = source
        
        if source == 'binance':
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
        
    def fetch_binance(self, symbol, timeframe='1h', start_date=None, end_date=None):
        """
        Fetch data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        
        # Convert dates to timestamps
        if start_date:
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        else:
            start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
        
        if end_date:
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)
        
        # Fetch OHLCV data
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter by end date
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df
    
    def fetch_yahoo(self, symbol, interval='1h', start_date=None, end_date=None):
        """
        Fetch data from Yahoo Finance (limited crypto pairs)
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: Candle interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=60)  # Yahoo limits historical data
        if not end_date:
            end_date = datetime.now()
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            interval=interval,
            start=start_date,
            end=end_date
        )
        
        # Rename columns to match our format
        df.columns = df.columns.str.lower()
        
        return df
    
    def fetch_coingecko(self, coin_id, vs_currency='usd', days=365):
        """
        Fetch data from CoinGecko API (daily data only)
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            vs_currency: Quote currency (e.g., 'usd')
            days: Number of days of historical data
        """
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume column (not available from CoinGecko OHLC)
            df['volume'] = 0
            
            return df
            
        except Exception as e:
            print(f"Error fetching from CoinGecko: {e}")
            return pd.DataFrame()
    
    def fetch_csv(self, filepath):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
        """
        
        df = pd.read_csv(filepath)
        
        # Ensure proper column names
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower()
        
        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def resample_timeframe(self, df, target_timeframe):
        """
        Resample data to a different timeframe
        
        Args:
            df: Original DataFrame
            target_timeframe: Target timeframe (e.g., '4H', '1D')
        """
        
        resampled = df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna()
    
    def add_synthetic_data(self, df, volatility=0.02):
        """
        Add synthetic future data for testing (useful for forward testing)
        
        Args:
            df: Original DataFrame
            volatility: Daily volatility for synthetic data
        """
        
        last_close = df['close'].iloc[-1]
        last_volume = df['volume'].mean()
        
        # Generate 30 days of synthetic data
        synthetic_data = []
        current_price = last_close
        
        for i in range(30 * 24):  # 30 days of hourly data
            # Random walk with trend
            returns = np.random.normal(0.0001, volatility/24)  # Slight upward bias
            current_price *= (1 + returns)
            
            # Generate OHLC
            open_price = current_price * (1 + np.random.normal(0, volatility/48))
            close_price = current_price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/48)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/48)))
            volume = last_volume * (1 + np.random.normal(0, 0.3))
            
            synthetic_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': max(0, volume)
            })
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df.index = pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=len(synthetic_data),
            freq='H'
        )
        
        # Combine with original data
        return pd.concat([df, synthetic_df])
    
    def validate_data(self, df):
        """
        Validate and clean the data
        
        Args:
            df: DataFrame to validate
        """
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Fill missing values
        df = df.fillna(method='ffill')
        
        # Ensure positive values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].abs()
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def get_data(self, symbol, **kwargs):
        """
        Main method to fetch data based on configured source
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional arguments for specific sources
        """
        
        if self.source == 'binance':
            df = self.fetch_binance(symbol, **kwargs)
        elif self.source == 'yahoo':
            df = self.fetch_yahoo(symbol, **kwargs)
        elif self.source == 'coingecko':
            df = self.fetch_coingecko(symbol, **kwargs)
        elif self.source == 'csv':
            df = self.fetch_csv(symbol)  # symbol is filepath in this case
        else:
            raise ValueError(f"Unknown data source: {self.source}")
        
        # Validate and clean data
        df = self.validate_data(df)
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = CryptoDataFetcher(source='binance')
    
    # Fetch BTC/USDT hourly data for the last 30 days
    df = fetcher.get_data(
        'BTC/USDT',
        timeframe='1h',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    print(f"Fetched {len(df)} candles")
    print(df.head())
    print(df.tail())