#!/usr/bin/env python3
"""
Download market data for multiple coins and timeframes
"""

import sys
import os
import time

# Add parent directory to path to import real_data_optimizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_optimizer import RealDataOptimizer

def download_all_data():
    """Download market data for all requested coins and timeframes"""
    
    coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'BNBUSDT', 'TRXUSDT', 'XRPUSDT']
    timeframes = ['15m', '1h', '4h', '1d']
    days = 180  # 90 days of historical data
    
    total_downloads = len(coins) * len(timeframes)
    current = 0
    
    print("=" * 80)
    print("DOWNLOADING MARKET DATA FOR MULTIPLE COINS")
    print("=" * 80)
    print(f"Coins: {', '.join(coins)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Days of history: {days}")
    print(f"Total downloads: {total_downloads}")
    print("=" * 80)
    print()
    
    results = []
    
    for coin in coins:
        for timeframe in timeframes:
            current += 1
            print(f"\n[{current}/{total_downloads}] Processing {coin} {timeframe}...")
            print("-" * 40)
            
            try:
                # Create optimizer instance (this will download/load data)
                optimizer = RealDataOptimizer(symbol=coin, timeframe=timeframe, days=days)
                
                # Download or load cached data
                candles = optimizer.download_binance_data()
                
                if candles:
                    file_size = os.path.getsize(
                        os.path.join(optimizer.data_dir, f"{coin}_{timeframe}_{days}days.json")
                    ) / (1024 * 1024)  # Convert to MB
                    
                    results.append({
                        'coin': coin,
                        'timeframe': timeframe,
                        'candles': len(candles),
                        'size_mb': round(file_size, 2),
                        'status': '✅ Success'
                    })
                    print(f"✅ Success: {len(candles)} candles, {file_size:.2f} MB")
                else:
                    results.append({
                        'coin': coin,
                        'timeframe': timeframe,
                        'candles': 0,
                        'size_mb': 0,
                        'status': '❌ Failed'
                    })
                    print(f"❌ Failed to download data")
                    
            except Exception as e:
                results.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'candles': 0,
                    'size_mb': 0,
                    'status': f'❌ Error: {str(e)[:50]}'
                })
                print(f"❌ Error: {e}")
            
            # Small delay between downloads to be nice to the API
            if current < total_downloads:
                time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"{'Coin':<10} {'Timeframe':<10} {'Candles':<10} {'Size (MB)':<10} {'Status':<20}")
    print("-" * 80)
    
    total_size = 0
    total_candles = 0
    successful = 0
    
    for result in results:
        print(f"{result['coin']:<10} {result['timeframe']:<10} {result['candles']:<10} "
              f"{result['size_mb']:<10} {result['status']:<20}")
        total_size += result['size_mb']
        total_candles += result['candles']
        if result['status'] == '✅ Success':
            successful += 1
    
    print("-" * 80)
    print(f"Total: {successful}/{total_downloads} successful, "
          f"{total_candles} candles, {total_size:.2f} MB")
    
    # List all cached files
    print("\n" + "=" * 80)
    print("CACHED FILES")
    print("=" * 80)
    
    data_dir = "backtest/market_data"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        files.sort()
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(data_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                mod_time = time.strftime('%Y-%m-%d %H:%M', 
                                         time.localtime(os.path.getmtime(file_path)))
                print(f"📁 {file:<40} {size_mb:>6.2f} MB   {mod_time}")
    
    print("\n✅ All downloads completed!")
    print("📊 You can now run backtests on any of these coins and timeframes.")
    print("\nExample commands:")
    print("  python3 backtest/real_data_optimizer.py ETHUSDT 15m 90")
    print("  python3 backtest/real_data_optimizer.py SOLUSDT 1h 90")

if __name__ == "__main__":
    download_all_data()