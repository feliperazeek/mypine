#!/usr/bin/env python3
"""
RSI Divergence Strategy Backtesting Script
Tests strategy across different parameter combinations and market data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import itertools
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from rsi_divergence_strategy import RSIDivergenceStrategy


class RSIDivergenceBacktest:
    """Backtest RSI Divergence Strategy with parameter optimization"""
    
    def __init__(self, data_file: str, initial_capital: float = 10000):
        """Initialize backtest"""
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.data = None
        self.results = []
        
    def load_data(self) -> bool:
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
            print(f"✓ Loaded {len(df)} candles from {Path(self.data_file).name}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def run_backtest(self, strategy: RSIDivergenceStrategy) -> Dict:
        """
        Run backtest with given strategy parameters
        Returns performance metrics
        """
        # Detect divergences and generate signals
        df = strategy.detect_divergences(self.data.copy())
        
        # Initialize tracking variables
        position = None  # None, 'long', or 'short'
        entry_price = 0
        entry_idx = 0
        stop_loss = None
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        position_size = 0
        
        # Commission
        commission_rate = 0.00045  # 0.045%
        
        # Simulate trading
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # Check exit conditions if in position
            if position == 'long':
                # Check stop loss
                if stop_loss is not None and current_row['low'] <= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'long',
                        'pnl': pnl,
                        'return_pct': (exit_price - entry_price) / entry_price * 100,
                        'exit_reason': 'stop_loss',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Check take profit (RSI crosses above threshold)
                elif current_row['rsi'] >= strategy.take_profit_rsi:
                    exit_price = current_row['close']
                    pnl = (exit_price - entry_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'long',
                        'pnl': pnl,
                        'return_pct': (exit_price - entry_price) / entry_price * 100,
                        'exit_reason': 'take_profit',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Check for opposite signal (bearish divergence)
                elif current_row['bearish_div'] or current_row['hidden_bearish_div']:
                    exit_price = current_row['close']
                    pnl = (exit_price - entry_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'long',
                        'pnl': pnl,
                        'return_pct': (exit_price - entry_price) / entry_price * 100,
                        'exit_reason': 'signal_exit',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Update trailing stop loss if using MIN_LOW
                elif strategy.sl_type == "MIN_LOW" and stop_loss is not None:
                    lookback_start = max(0, i - strategy.min_low_lookback)
                    new_stop = df['low'].iloc[lookback_start:i+1].min()
                    stop_loss = max(stop_loss, new_stop)
            
            elif position == 'short':
                # Check stop loss
                if stop_loss is not None and current_row['high'] >= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'short',
                        'pnl': pnl,
                        'return_pct': (entry_price - exit_price) / entry_price * 100,
                        'exit_reason': 'stop_loss',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Check take profit (RSI crosses below threshold)
                elif current_row['rsi'] <= (100 - strategy.take_profit_rsi):
                    exit_price = current_row['close']
                    pnl = (entry_price - exit_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'short',
                        'pnl': pnl,
                        'return_pct': (entry_price - exit_price) / entry_price * 100,
                        'exit_reason': 'take_profit',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Check for opposite signal (bullish divergence)
                elif current_row['bullish_div'] or current_row['hidden_bullish_div']:
                    exit_price = current_row['close']
                    pnl = (entry_price - exit_price) * position_size
                    commission = (entry_price * position_size + exit_price * position_size) * commission_rate
                    pnl -= commission
                    
                    trades.append({
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'short',
                        'pnl': pnl,
                        'return_pct': (entry_price - exit_price) / entry_price * 100,
                        'exit_reason': 'signal_exit',
                        'bars_held': i - entry_idx
                    })
                    
                    current_capital += pnl
                    position = None
                    position_size = 0
                    stop_loss = None
                
                # Update trailing stop loss if using MIN_LOW
                elif strategy.sl_type == "MIN_LOW" and stop_loss is not None:
                    lookback_start = max(0, i - strategy.min_low_lookback)
                    new_stop = df['high'].iloc[lookback_start:i+1].max()
                    stop_loss = min(stop_loss, new_stop)
            
            # Check entry conditions if not in position
            if position is None:
                # Long entry
                if current_row['long_signal']:
                    entry_price = current_row['close']
                    position_size = current_capital * 1.0 / entry_price  # 100% of capital
                    entry_idx = i
                    position = 'long'
                    stop_loss = strategy.calculate_stop_loss(df, i, 'long')
                
                # Short entry
                elif current_row['short_signal']:
                    entry_price = current_row['close']
                    position_size = current_capital * 1.0 / entry_price  # 100% of capital
                    entry_idx = i
                    position = 'short'
                    stop_loss = strategy.calculate_stop_loss(df, i, 'short')
            
            # Update equity curve
            if position == 'long':
                unrealized_pnl = (current_row['close'] - entry_price) * position_size
                equity_curve.append(current_capital + unrealized_pnl)
            elif position == 'short':
                unrealized_pnl = (entry_price - current_row['close']) * position_size
                equity_curve.append(current_capital + unrealized_pnl)
            else:
                equity_curve.append(current_capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity_curve, strategy.get_strategy_params())
        return metrics
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float], params: Dict) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'params': params,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_bars_held': 0,
                'sharpe_ratio': 0,
                'final_capital': self.initial_capital
            }
        
        # Basic stats
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown calculation
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Returns calculation for Sharpe ratio
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0
        
        final_capital = equity_curve[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'params': params,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'total_pnl': sum(t['pnl'] for t in trades),
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'avg_bars_held': np.mean([t['bars_held'] for t in trades]),
            'sharpe_ratio': sharpe_ratio,
            'final_capital': final_capital
        }
    
    def optimize_parameters(self) -> List[Dict]:
        """
        Test multiple parameter combinations
        Returns list of results sorted by performance
        """
        print("\n" + "="*80)
        print("OPTIMIZING RSI DIVERGENCE STRATEGY PARAMETERS")
        print("="*80)
        
        # Define parameter ranges to test
        param_grid = {
            'rsi_length': [5, 7, 9, 11, 14],
            'pivot_left': [1, 2, 3],
            'pivot_right': [1, 2, 3, 5],
            'take_profit_rsi': [70, 75, 80, 85],
            'sl_type': ['NONE', 'PERC', 'ATR'],
            'stop_loss_percent': [3.0, 5.0, 7.0],
            'atr_multiplier': [2.0, 3.0, 3.5, 4.0],
        }
        
        # Generate all combinations (limited for performance)
        # We'll test a subset to keep runtime reasonable
        test_configs = []
        
        # Strategy 1: Long only, regular bullish divergences
        for rsi_len in param_grid['rsi_length']:
            for pl in param_grid['pivot_left']:
                for pr in param_grid['pivot_right']:
                    for tp_rsi in param_grid['take_profit_rsi']:
                        for sl_type in param_grid['sl_type']:
                            if sl_type == 'PERC':
                                for sl_pct in param_grid['stop_loss_percent']:
                                    test_configs.append({
                                        'rsi_length': rsi_len,
                                        'pivot_left': pl,
                                        'pivot_right': pr,
                                        'take_profit_rsi': tp_rsi,
                                        'sl_type': sl_type,
                                        'stop_loss_percent': sl_pct,
                                        'enable_long': True,
                                        'enable_short': False,
                                        'plot_regular_bull': True,
                                        'plot_hidden_bull': False,
                                        'plot_regular_bear': True,
                                        'plot_hidden_bear': False
                                    })
                            elif sl_type == 'ATR':
                                for atr_mult in param_grid['atr_multiplier']:
                                    test_configs.append({
                                        'rsi_length': rsi_len,
                                        'pivot_left': pl,
                                        'pivot_right': pr,
                                        'take_profit_rsi': tp_rsi,
                                        'sl_type': sl_type,
                                        'atr_multiplier': atr_mult,
                                        'enable_long': True,
                                        'enable_short': False,
                                        'plot_regular_bull': True,
                                        'plot_hidden_bull': False,
                                        'plot_regular_bear': True,
                                        'plot_hidden_bear': False
                                    })
                            else:  # NONE
                                test_configs.append({
                                    'rsi_length': rsi_len,
                                    'pivot_left': pl,
                                    'pivot_right': pr,
                                    'take_profit_rsi': tp_rsi,
                                    'sl_type': sl_type,
                                    'enable_long': True,
                                    'enable_short': False,
                                    'plot_regular_bull': True,
                                    'plot_hidden_bull': False,
                                    'plot_regular_bear': True,
                                    'plot_hidden_bear': False
                                })
        
        # Also test with hidden divergences
        for rsi_len in [5, 9, 14]:
            for pr in [3, 5]:
                for tp_rsi in [75, 80]:
                    test_configs.append({
                        'rsi_length': rsi_len,
                        'pivot_left': 1,
                        'pivot_right': pr,
                        'take_profit_rsi': tp_rsi,
                        'sl_type': 'NONE',
                        'enable_long': True,
                        'enable_short': False,
                        'plot_regular_bull': True,
                        'plot_hidden_bull': True,
                        'plot_regular_bear': True,
                        'plot_hidden_bear': False
                    })
        
        print(f"\nTesting {len(test_configs)} parameter combinations...")
        print(f"Data file: {Path(self.data_file).name}")
        
        results = []
        for idx, config in enumerate(test_configs):
            if (idx + 1) % 100 == 0:
                print(f"Progress: {idx + 1}/{len(test_configs)} configurations tested...")
            
            strategy = RSIDivergenceStrategy(**config)
            metrics = self.run_backtest(strategy)
            results.append(metrics)
        
        self.results = results
        return results


def run_optimization():
    """Run optimization across all market data files"""
    data_dir = Path("/Users/foliveira/mypine/backtest/market_data")
    data_files = sorted(data_dir.glob("*.json"))
    
    all_results = {}
    
    print("\n" + "="*80)
    print("RSI DIVERGENCE STRATEGY - COMPREHENSIVE BACKTEST")
    print("="*80)
    print(f"\nFound {len(data_files)} market data files")
    
    for data_file in data_files:
        pair_name = data_file.stem  # e.g., "BTCUSDT_1h_90days"
        
        print(f"\n{'='*80}")
        print(f"TESTING: {pair_name}")
        print(f"{'='*80}")
        
        backtest = RSIDivergenceBacktest(str(data_file))
        if not backtest.load_data():
            continue
        
        results = backtest.optimize_parameters()
        all_results[pair_name] = results
    
    return all_results


def print_report(all_results: Dict):
    """Print comprehensive report of results"""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS - SORTED BY PROFIT FACTOR AND DRAWDOWN")
    print("="*80)
    
    for pair_name, results in all_results.items():
        print(f"\n{'='*80}")
        print(f"PAIR: {pair_name}")
        print(f"{'='*80}")
        
        # Filter out results with no trades or negative returns
        valid_results = [r for r in results if r['total_trades'] > 5 and r['total_return'] > 0]
        
        if not valid_results:
            print("No profitable configurations found with minimum 5 trades.")
            continue
        
        # Sort by profit factor (descending) and max drawdown (ascending)
        sorted_results = sorted(
            valid_results,
            key=lambda x: (x['profit_factor'], -x['max_drawdown_pct']),
            reverse=True
        )
        
        # Show top 10 configurations
        print(f"\nTop 10 Configurations (by Profit Factor, lowest Drawdown):")
        print("-" * 80)
        
        for idx, result in enumerate(sorted_results[:10], 1):
            params = result['params']
            print(f"\n#{idx}:")
            print(f"  RSI Length: {params['rsi_length']} | "
                  f"Pivot L/R: {params['pivot_left']}/{params['pivot_right']} | "
                  f"TP RSI: {params['take_profit_rsi']} | "
                  f"SL Type: {params['sl_type']}")
            
            if params['sl_type'] == 'PERC':
                print(f"  Stop Loss %: {params['stop_loss_percent']:.1f}")
            elif params['sl_type'] == 'ATR':
                print(f"  ATR Multiplier: {params['atr_multiplier']:.1f}")
            
            print(f"  Divergences: Regular Bull: {params['plot_regular_bull']}, "
                  f"Hidden Bull: {params['plot_hidden_bull']}")
            
            print(f"  → Total Trades: {result['total_trades']}")
            print(f"  → Win Rate: {result['win_rate']:.2f}%")
            print(f"  → Total Return: {result['total_return']:.2f}%")
            print(f"  → Profit Factor: {result['profit_factor']:.2f}")
            print(f"  → Max Drawdown: {result['max_drawdown_pct']:.2f}%")
            print(f"  → Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  → Avg Bars Held: {result['avg_bars_held']:.1f}")
            print(f"  → Final Capital: ${result['final_capital']:.2f}")
    
    # Generate summary CSV
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORTS")
    print("="*80)
    
    output_dir = Path("/Users/foliveira/mypine/backtest/rsi-div")
    
    # Create detailed CSV for each pair
    for pair_name, results in all_results.items():
        valid_results = [r for r in results if r['total_trades'] > 5]
        
        if not valid_results:
            continue
        
        # Convert to DataFrame
        rows = []
        for result in valid_results:
            row = {
                'pair': pair_name,
                'rsi_length': result['params']['rsi_length'],
                'pivot_left': result['params']['pivot_left'],
                'pivot_right': result['params']['pivot_right'],
                'take_profit_rsi': result['params']['take_profit_rsi'],
                'sl_type': result['params']['sl_type'],
                'total_trades': result['total_trades'],
                'win_rate': result['win_rate'],
                'total_return': result['total_return'],
                'profit_factor': result['profit_factor'],
                'max_drawdown_pct': result['max_drawdown_pct'],
                'sharpe_ratio': result['sharpe_ratio'],
                'final_capital': result['final_capital']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['profit_factor', 'max_drawdown_pct'], ascending=[False, True])
        
        csv_file = output_dir / f"results_{pair_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    all_results = run_optimization()
    print_report(all_results)

