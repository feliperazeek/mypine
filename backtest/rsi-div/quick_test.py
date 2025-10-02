#!/usr/bin/env python3
"""
Quick test of RSI Divergence Strategy
Tests one configuration on one data file to verify implementation
"""

import json
import pandas as pd
from pathlib import Path
from rsi_divergence_strategy import RSIDivergenceStrategy
from backtest import RSIDivergenceBacktest


def quick_test():
    """Run a quick test with default parameters"""
    print("\n" + "="*80)
    print("RSI DIVERGENCE STRATEGY - QUICK TEST")
    print("="*80)
    
    # Test with one data file
    data_file = "/Users/foliveira/mypine/backtest/market_data/BTCUSDT_1h_180days.json"
    
    print(f"\nTesting with: {Path(data_file).name}")
    print(f"Using default strategy parameters...")
    
    # Create backtest instance
    backtest = RSIDivergenceBacktest(data_file)
    
    if not backtest.load_data():
        return
    
    # Test with default parameters (optimized settings from Pine Script comments)
    configs_to_test = [
        {
            'name': 'Default (RSI 9, Pivot 1/3, TP 80)',
            'params': {
                'rsi_length': 9,
                'pivot_left': 1,
                'pivot_right': 3,
                'take_profit_rsi': 80,
                'sl_type': 'NONE',
                'enable_long': True,
                'enable_short': False,
                'plot_regular_bull': True,
                'plot_hidden_bull': True,
                'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        },
        {
            'name': 'GOOGL Optimized (RSI 5, Pivot 1/3, TP 75)',
            'params': {
                'rsi_length': 5,
                'pivot_left': 1,
                'pivot_right': 3,
                'take_profit_rsi': 75,
                'sl_type': 'NONE',
                'enable_long': True,
                'enable_short': False,
                'plot_regular_bull': True,
                'plot_hidden_bull': False,
                'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        },
        {
            'name': 'With 5% Stop Loss',
            'params': {
                'rsi_length': 9,
                'pivot_left': 1,
                'pivot_right': 3,
                'take_profit_rsi': 80,
                'sl_type': 'PERC',
                'stop_loss_percent': 5.0,
                'enable_long': True,
                'enable_short': False,
                'plot_regular_bull': True,
                'plot_hidden_bull': True,
                'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        },
        {
            'name': 'With ATR Stop Loss',
            'params': {
                'rsi_length': 9,
                'pivot_left': 1,
                'pivot_right': 3,
                'take_profit_rsi': 80,
                'sl_type': 'ATR',
                'atr_multiplier': 3.5,
                'enable_long': True,
                'enable_short': False,
                'plot_regular_bull': True,
                'plot_hidden_bull': True,
                'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        }
    ]
    
    print("\n" + "-"*80)
    print("TESTING CONFIGURATIONS")
    print("-"*80)
    
    results = []
    
    for config in configs_to_test:
        print(f"\nTesting: {config['name']}")
        
        strategy = RSIDivergenceStrategy(**config['params'])
        metrics = backtest.run_backtest(strategy)
        
        results.append({
            'name': config['name'],
            'metrics': metrics
        })
        
        print(f"  → Total Trades: {metrics['total_trades']}")
        print(f"  → Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  → Total Return: {metrics['total_return']:.2f}%")
        print(f"  → Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  → Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  → Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  → Final Capital: ${metrics['final_capital']:.2f}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Sort by total return
    results_sorted = sorted(results, key=lambda x: x['metrics']['total_return'], reverse=True)
    
    print("\nRanked by Total Return:")
    for idx, result in enumerate(results_sorted, 1):
        metrics = result['metrics']
        print(f"\n#{idx}: {result['name']}")
        print(f"  Return: {metrics['total_return']:.2f}% | "
              f"PF: {metrics['profit_factor']:.2f} | "
              f"Trades: {metrics['total_trades']} | "
              f"Win Rate: {metrics['win_rate']:.2f}%")
    
    print("\n✓ Quick test completed successfully!")
    print("\nTo run full optimization across all market data, use: python backtest.py")


if __name__ == "__main__":
    quick_test()

