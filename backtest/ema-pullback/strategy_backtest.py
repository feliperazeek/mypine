#!/usr/bin/env python3
"""
Strategy Backtest Comparison Tool
Compares EMA20 Pullback vs Hotzone Pullback strategies
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktest:
    def __init__(self, data_file, strategy_name):
        self.data_file = data_file
        self.strategy_name = strategy_name
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
    def load_data(self):
        """Load market data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            self.data = df
            print(f"Loaded {len(df)} candles for {self.strategy_name}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_ema(self, period):
        """Calculate EMA for given period"""
        return self.data['close'].ewm(span=period).mean()
    
    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_volume_sma(self, period=20):
        """Calculate volume SMA"""
        return self.data['volume'].rolling(window=period).mean()
    
    def detect_liquidation_zones(self, length=4):
        """Detect liquidation zones based on volume spikes"""
        zones = []
        
        # Calculate pivot high volume
        phv = self.data['volume'].rolling(window=length*2+1, center=True).max() == self.data['volume']
        
        # Find bullish zones (high volume at lows)
        for i in range(length, len(self.data) - length):
            if phv.iloc[i]:
                # Check if this is a low with high volume
                window_low = self.data['low'].iloc[i-length:i+length+1].min()
                if self.data['low'].iloc[i] == window_low:
                    zones.append({
                        'type': 'bullish',
                        'top': self.data['high'].iloc[i],
                        'bottom': self.data['low'].iloc[i],
                        'time': self.data.index[i],
                        'volume': self.data['volume'].iloc[i]
                    })
        
        # Find bearish zones (high volume at highs)
        for i in range(length, len(self.data) - length):
            if phv.iloc[i]:
                # Check if this is a high with high volume
                window_high = self.data['high'].iloc[i-length:i+length+1].max()
                if self.data['high'].iloc[i] == window_high:
                    zones.append({
                        'type': 'bearish',
                        'top': self.data['high'].iloc[i],
                        'bottom': self.data['low'].iloc[i],
                        'time': self.data.index[i],
                        'volume': self.data['volume'].iloc[i]
                    })
        
        return zones
    
    def is_in_zone(self, price, zones, zone_type):
        """Check if price is in a specific type of zone"""
        for zone in zones:
            if zone['type'] == zone_type:
                if zone['bottom'] <= price <= zone['top']:
                    return True
        return False
    
    def get_closest_zone(self, price, zones, zone_type):
        """Get closest zone of specific type"""
        closest = None
        min_distance = float('inf')
        
        for zone in zones:
            if zone['type'] == zone_type:
                if zone_type == 'bullish' and price < zone['bottom']:
                    distance = zone['bottom'] - price
                    if distance < min_distance:
                        min_distance = distance
                        closest = zone['bottom']
                elif zone_type == 'bearish' and price > zone['top']:
                    distance = price - zone['top']
                    if distance < min_distance:
                        min_distance = distance
                        closest = zone['top']
        
        return closest

class EMAPullbackStrategy(StrategyBacktest):
    def __init__(self, data_file):
        super().__init__(data_file, "EMA20 Pullback")
        self.ema20_period = 20
        self.ema200_period = 200
        self.pullback_pct = 0.1
        self.min_candle_size = 0.5
        self.sl_percent = 1.0
        self.rr_ratio = 2.0
        
    def run_backtest(self):
        """Run EMA20 pullback strategy backtest"""
        if not self.load_data():
            return False
        
        # Calculate indicators
        self.data['ema20'] = self.calculate_ema(self.ema20_period)
        self.data['ema200'] = self.calculate_ema(self.ema200_period)
        self.data['rsi'] = self.calculate_rsi()
        self.data['volume_sma'] = self.calculate_volume_sma()
        
        # Initialize equity curve
        self.equity_curve = [self.initial_capital]
        
        for i in range(200, len(self.data)):  # Start after EMA200 is calculated
            current = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            
            # Check for long entry
            if (self.position == 0 and 
                current['close'] > current['open'] and  # Bullish candle
                current['low'] <= current['ema20'] and  # Touched EMA20
                current['close'] > current['ema20'] and  # Closed above EMA20
                current['ema20'] > current['ema200'] and  # Uptrend
                current['volume'] > current['volume_sma'] * 1.5):  # Volume filter
                
                # Calculate pullback
                pullback = (current['high'] - current['ema20']) / current['ema20'] * 100
                candle_size = abs(current['close'] - current['open']) / current['open'] * 100
                
                if pullback >= self.pullback_pct and candle_size >= self.min_candle_size:
                    self.enter_long(current, i)
            
            # Check for short entry
            elif (self.position == 0 and 
                  current['close'] < current['open'] and  # Bearish candle
                  current['high'] >= current['ema20'] and  # Touched EMA20
                  current['close'] < current['ema20'] and  # Closed below EMA20
                  current['ema20'] < current['ema200'] and  # Downtrend
                  current['volume'] > current['volume_sma'] * 1.5):  # Volume filter
                
                # Calculate pullback
                pullback = (current['ema20'] - current['low']) / current['ema20'] * 100
                candle_size = abs(current['close'] - current['open']) / current['open'] * 100
                
                if pullback >= self.pullback_pct and candle_size >= self.min_candle_size:
                    self.enter_short(current, i)
            
            # Check for exits
            if self.position != 0:
                self.check_exit(current, i)
            
            # Update equity curve
            if self.position != 0:
                unrealized_pnl = (current['close'] - self.entry_price) * self.position_size
                self.current_capital = self.initial_capital + sum([trade['pnl'] for trade in self.trades]) + unrealized_pnl
            else:
                self.current_capital = self.initial_capital + sum([trade['pnl'] for trade in self.trades])
            
            self.equity_curve.append(self.current_capital)
        
        return True
    
    def enter_long(self, current, index):
        """Enter long position"""
        self.position = 1
        self.entry_price = current['close']
        self.position_size = self.current_capital * 0.1 / self.entry_price  # 10% of capital
        self.stop_loss = self.entry_price * (1 - self.sl_percent / 100)
        self.take_profit = self.entry_price + (self.entry_price - self.stop_loss) * self.rr_ratio
        
    def enter_short(self, current, index):
        """Enter short position"""
        self.position = -1
        self.entry_price = current['close']
        self.position_size = self.current_capital * 0.1 / self.entry_price  # 10% of capital
        self.stop_loss = self.entry_price * (1 + self.sl_percent / 100)
        self.take_profit = self.entry_price - (self.stop_loss - self.entry_price) * self.rr_ratio
    
    def check_exit(self, current, index):
        """Check for exit conditions"""
        if self.position == 1:  # Long position
            if current['low'] <= self.stop_loss:
                self.exit_position(current, "Stop Loss")
            elif current['high'] >= self.take_profit:
                self.exit_position(current, "Take Profit")
        elif self.position == -1:  # Short position
            if current['high'] >= self.stop_loss:
                self.exit_position(current, "Stop Loss")
            elif current['low'] <= self.take_profit:
                self.exit_position(current, "Take Profit")
    
    def exit_position(self, current, reason):
        """Exit current position"""
        if self.position == 1:  # Long
            pnl = (current['close'] - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - current['close']) * self.position_size
        
        self.trades.append({
            'entry_time': self.data.index[len(self.trades)],
            'exit_time': current.name,
            'entry_price': self.entry_price,
            'exit_price': current['close'],
            'position_size': self.position_size,
            'pnl': pnl,
            'reason': reason
        })
        
        self.position = 0
        self.position_size = 0

class HotzonePullbackStrategy(StrategyBacktest):
    def __init__(self, data_file):
        super().__init__(data_file, "Hotzone Pullback")
        self.liquidity_length = 4
        self.pullback_pct = 0.5
        self.min_candle_size = 0.5
        self.sl_percent = 1.0
        self.rr_ratio = 2.0
        self.zones = []
        
    def run_backtest(self):
        """Run Hotzone pullback strategy backtest"""
        if not self.load_data():
            return False
        
        # Calculate indicators
        self.data['volume_sma'] = self.calculate_volume_sma()
        self.zones = self.detect_liquidation_zones(self.liquidity_length)
        
        # Initialize equity curve
        self.equity_curve = [self.initial_capital]
        
        for i in range(50, len(self.data)):  # Start after enough data for zones
            current = self.data.iloc[i]
            
            # Check for long entry (pullback into bullish zone)
            if (self.position == 0 and 
                current['close'] > current['open'] and  # Bullish candle
                self.is_in_zone(current['close'], self.zones, 'bullish') and  # In bullish zone
                current['volume'] > current['volume_sma'] * 1.5):  # Volume filter
                
                # Calculate pullback
                pullback = (current['high'] - current['low']) / current['low'] * 100
                candle_size = abs(current['close'] - current['open']) / current['open'] * 100
                
                if pullback >= self.pullback_pct and candle_size >= self.min_candle_size:
                    self.enter_long(current, i)
            
            # Check for short entry (pullback into bearish zone)
            elif (self.position == 0 and 
                  current['close'] < current['open'] and  # Bearish candle
                  self.is_in_zone(current['close'], self.zones, 'bearish') and  # In bearish zone
                  current['volume'] > current['volume_sma'] * 1.5):  # Volume filter
                
                # Calculate pullback
                pullback = (current['high'] - current['low']) / current['low'] * 100
                candle_size = abs(current['close'] - current['open']) / current['open'] * 100
                
                if pullback >= self.pullback_pct and candle_size >= self.min_candle_size:
                    self.enter_short(current, i)
            
            # Check for exits
            if self.position != 0:
                self.check_exit(current, i)
            
            # Update equity curve
            if self.position != 0:
                unrealized_pnl = (current['close'] - self.entry_price) * self.position_size
                self.current_capital = self.initial_capital + sum([trade['pnl'] for trade in self.trades]) + unrealized_pnl
            else:
                self.current_capital = self.initial_capital + sum([trade['pnl'] for trade in self.trades])
            
            self.equity_curve.append(self.current_capital)
        
        return True
    
    def enter_long(self, current, index):
        """Enter long position"""
        self.position = 1
        self.entry_price = current['close']
        self.position_size = self.current_capital * 0.1 / self.entry_price  # 10% of capital
        self.stop_loss = self.entry_price * (1 - self.sl_percent / 100)
        
        # Try to find next bullish zone for take profit
        next_zone = self.get_closest_zone(current['close'], self.zones, 'bullish')
        if next_zone:
            self.take_profit = next_zone
        else:
            self.take_profit = self.entry_price + (self.entry_price - self.stop_loss) * self.rr_ratio
        
    def enter_short(self, current, index):
        """Enter short position"""
        self.position = -1
        self.entry_price = current['close']
        self.position_size = self.current_capital * 0.1 / self.entry_price  # 10% of capital
        self.stop_loss = self.entry_price * (1 + self.sl_percent / 100)
        
        # Try to find next bearish zone for take profit
        next_zone = self.get_closest_zone(current['close'], self.zones, 'bearish')
        if next_zone:
            self.take_profit = next_zone
        else:
            self.take_profit = self.entry_price - (self.stop_loss - self.entry_price) * self.rr_ratio
    
    def check_exit(self, current, index):
        """Check for exit conditions"""
        if self.position == 1:  # Long position
            if current['low'] <= self.stop_loss:
                self.exit_position(current, "Stop Loss")
            elif current['high'] >= self.take_profit:
                self.exit_position(current, "Take Profit")
        elif self.position == -1:  # Short position
            if current['high'] >= self.stop_loss:
                self.exit_position(current, "Stop Loss")
            elif current['low'] <= self.take_profit:
                self.exit_position(current, "Take Profit")
    
    def exit_position(self, current, reason):
        """Exit current position"""
        if self.position == 1:  # Long
            pnl = (current['close'] - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - current['close']) * self.position_size
        
        self.trades.append({
            'entry_time': self.data.index[len(self.trades)],
            'exit_time': current.name,
            'entry_price': self.entry_price,
            'exit_price': current['close'],
            'position_size': self.position_size,
            'pnl': pnl,
            'reason': reason
        })
        
        self.position = 0
        self.position_size = 0

def run_comparison():
    """Run comparison between both strategies"""
    data_dir = "/Users/foliveira/mypine/backtest/market_data"
    results = {}
    
    # Get all available data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    print("Running strategy comparison...")
    print("=" * 50)
    
    for data_file in data_files:
        if '1h' not in data_file:  # Focus on 1-hour data for now
            continue
            
        file_path = os.path.join(data_dir, data_file)
        pair_name = data_file.replace('_1h_90days.json', '')
        
        print(f"\nTesting {pair_name}...")
        
        # Run EMA strategy
        ema_strategy = EMAPullbackStrategy(file_path)
        ema_success = ema_strategy.run_backtest()
        
        # Run Hotzone strategy
        hotzone_strategy = HotzonePullbackStrategy(file_path)
        hotzone_success = hotzone_strategy.run_backtest()
        
        if ema_success and hotzone_success:
            results[pair_name] = {
                'ema': {
                    'total_trades': len(ema_strategy.trades),
                    'winning_trades': len([t for t in ema_strategy.trades if t['pnl'] > 0]),
                    'losing_trades': len([t for t in ema_strategy.trades if t['pnl'] < 0]),
                    'total_pnl': sum([t['pnl'] for t in ema_strategy.trades]),
                    'final_capital': ema_strategy.current_capital,
                    'return_pct': (ema_strategy.current_capital - ema_strategy.initial_capital) / ema_strategy.initial_capital * 100,
                    'max_drawdown': calculate_max_drawdown(ema_strategy.equity_curve),
                    'win_rate': len([t for t in ema_strategy.trades if t['pnl'] > 0]) / len(ema_strategy.trades) * 100 if ema_strategy.trades else 0
                },
                'hotzone': {
                    'total_trades': len(hotzone_strategy.trades),
                    'winning_trades': len([t for t in hotzone_strategy.trades if t['pnl'] > 0]),
                    'losing_trades': len([t for t in hotzone_strategy.trades if t['pnl'] < 0]),
                    'total_pnl': sum([t['pnl'] for t in hotzone_strategy.trades]),
                    'final_capital': hotzone_strategy.current_capital,
                    'return_pct': (hotzone_strategy.current_capital - hotzone_strategy.initial_capital) / hotzone_strategy.initial_capital * 100,
                    'max_drawdown': calculate_max_drawdown(hotzone_strategy.equity_curve),
                    'win_rate': len([t for t in hotzone_strategy.trades if t['pnl'] > 0]) / len(hotzone_strategy.trades) * 100 if hotzone_strategy.trades else 0
                }
            }
    
    return results

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

def print_results(results):
    """Print comparison results"""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 80)
    
    for pair, data in results.items():
        print(f"\n{pair}:")
        print("-" * 40)
        
        ema = data['ema']
        hotzone = data['hotzone']
        
        print(f"{'Metric':<20} {'EMA20 Pullback':<15} {'Hotzone Pullback':<15} {'Winner':<10}")
        print("-" * 70)
        
        # Total Trades
        ema_trades = ema['total_trades']
        hz_trades = hotzone['total_trades']
        trades_winner = "EMA" if ema_trades > hz_trades else "Hotzone" if hz_trades > ema_trades else "Tie"
        print(f"{'Total Trades':<20} {ema_trades:<15} {hz_trades:<15} {trades_winner:<10}")
        
        # Win Rate
        ema_wr = f"{ema['win_rate']:.1f}%"
        hz_wr = f"{hotzone['win_rate']:.1f}%"
        wr_winner = "EMA" if ema['win_rate'] > hotzone['win_rate'] else "Hotzone" if hotzone['win_rate'] > ema['win_rate'] else "Tie"
        print(f"{'Win Rate':<20} {ema_wr:<15} {hz_wr:<15} {wr_winner:<10}")
        
        # Total Return
        ema_ret = f"{ema['return_pct']:.2f}%"
        hz_ret = f"{hotzone['return_pct']:.2f}%"
        ret_winner = "EMA" if ema['return_pct'] > hotzone['return_pct'] else "Hotzone" if hotzone['return_pct'] > ema['return_pct'] else "Tie"
        print(f"{'Total Return':<20} {ema_ret:<15} {hz_ret:<15} {ret_winner:<10}")
        
        # Max Drawdown
        ema_dd = f"{ema['max_drawdown']:.2f}%"
        hz_dd = f"{hotzone['max_drawdown']:.2f}%"
        dd_winner = "EMA" if ema['max_drawdown'] < hotzone['max_drawdown'] else "Hotzone" if hotzone['max_drawdown'] < ema['max_drawdown'] else "Tie"
        print(f"{'Max Drawdown':<20} {ema_dd:<15} {hz_dd:<15} {dd_winner:<10}")
        
        # Final Capital
        ema_cap = f"${ema['final_capital']:,.0f}"
        hz_cap = f"${hotzone['final_capital']:,.0f}"
        cap_winner = "EMA" if ema['final_capital'] > hotzone['final_capital'] else "Hotzone" if hotzone['final_capital'] > ema['final_capital'] else "Tie"
        print(f"{'Final Capital':<20} {ema_cap:<15} {hz_cap:<15} {cap_winner:<10}")

if __name__ == "__main__":
    results = run_comparison()
    print_results(results)

