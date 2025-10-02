#!/usr/bin/env python3
"""
Fast RSI Divergence Strategy Backtesting
Tests curated parameter combinations for faster results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from rsi_divergence_strategy import RSIDivergenceStrategy
from backtest import RSIDivergenceBacktest


def run_fast_optimization():
    """Run optimization with curated parameter combinations"""
    data_dir = Path("/Users/foliveira/mypine/backtest/market_data")
    data_files = sorted(data_dir.glob("*.json"))
    
    # Curated parameter combinations based on common profitable settings
    test_configs = []
    
    # Best performing RSI lengths
    rsi_lengths = [5, 7, 9, 14]
    
    # Effective pivot combinations
    pivot_combos = [
        (1, 3),
        (1, 5),
        (2, 3),
        (3, 3)
    ]
    
    # Take profit levels
    tp_levels = [70, 75, 80, 85]
    
    # Stop loss configurations
    sl_configs = [
        {'sl_type': 'NONE'},
        {'sl_type': 'PERC', 'stop_loss_percent': 3.0},
        {'sl_type': 'PERC', 'stop_loss_percent': 5.0},
        {'sl_type': 'ATR', 'atr_multiplier': 2.5},
        {'sl_type': 'ATR', 'atr_multiplier': 3.5}
    ]
    
    # Divergence type combinations
    div_configs = [
        {'regular': True, 'hidden': False},   # Regular only
        {'regular': True, 'hidden': True},    # Both
        {'regular': False, 'hidden': True}    # Hidden only
    ]
    
    # Generate configurations
    for rsi_len in rsi_lengths:
        for pivot_left, pivot_right in pivot_combos:
            for tp_rsi in tp_levels:
                for sl_config in sl_configs:
                    for div_config in div_configs:
                        config = {
                            'rsi_length': rsi_len,
                            'pivot_left': pivot_left,
                            'pivot_right': pivot_right,
                            'take_profit_rsi': tp_rsi,
                            'enable_long': True,
                            'enable_short': False,
                            'plot_regular_bull': div_config['regular'],
                            'plot_hidden_bull': div_config['hidden'],
                            'plot_regular_bear': True,
                            'plot_hidden_bear': False,
                            **sl_config
                        }
                        test_configs.append(config)
    
    print("\n" + "="*80)
    print("RSI DIVERGENCE STRATEGY - FAST OPTIMIZATION")
    print("="*80)
    print(f"\nTotal configurations to test: {len(test_configs)}")
    print(f"Market data files: {len(data_files)}")
    print(f"Estimated time: {len(test_configs) * len(data_files) * 0.5 / 60:.1f} minutes\n")
    
    all_results = {}
    
    for data_file in data_files:
        pair_name = data_file.stem
        
        print(f"\n{'='*80}")
        print(f"TESTING: {pair_name}")
        print(f"{'='*80}")
        
        backtest = RSIDivergenceBacktest(str(data_file))
        if not backtest.load_data():
            continue
        
        results = []
        for idx, config in enumerate(test_configs):
            if (idx + 1) % 50 == 0:
                progress_pct = (idx + 1) / len(test_configs) * 100
                print(f"Progress: {idx + 1}/{len(test_configs)} ({progress_pct:.1f}%)...")
            
            strategy = RSIDivergenceStrategy(**config)
            metrics = backtest.run_backtest(strategy)
            results.append(metrics)
        
        all_results[pair_name] = results
        
        # Show top 5 results immediately
        valid_results = [r for r in results if r['total_trades'] >= 5 and r['total_return'] > 0]
        if valid_results:
            sorted_results = sorted(
                valid_results,
                key=lambda x: (x['profit_factor'], -x['max_drawdown_pct']),
                reverse=True
            )
            
            print(f"\nTop 5 configurations for {pair_name}:")
            print("-" * 80)
            for idx, result in enumerate(sorted_results[:5], 1):
                params = result['params']
                print(f"#{idx}: RSI{params['rsi_length']} P{params['pivot_left']}/{params['pivot_right']} "
                      f"TP{params['take_profit_rsi']} {params['sl_type']} → "
                      f"Return: {result['total_return']:.2f}% | "
                      f"PF: {result['profit_factor']:.2f} | "
                      f"WR: {result['win_rate']:.1f}% | "
                      f"DD: {result['max_drawdown_pct']:.2f}%")
    
    return all_results


def print_comprehensive_report(all_results: Dict):
    """Print detailed report across all markets"""
    print("\n" + "="*80)
    print("COMPREHENSIVE OPTIMIZATION REPORT")
    print("="*80)
    
    output_dir = Path("/Users/foliveira/mypine/backtest/rsi-div")
    
    # Summary statistics
    summary_rows = []
    
    for pair_name, results in all_results.items():
        valid_results = [r for r in results if r['total_trades'] >= 5]
        
        if not valid_results:
            continue
        
        # Best by profit factor
        best_pf = max(valid_results, key=lambda x: x['profit_factor'])
        # Best by return
        best_return = max(valid_results, key=lambda x: x['total_return'])
        # Best by win rate
        best_wr = max(valid_results, key=lambda x: x['win_rate'])
        
        summary_rows.append({
            'pair': pair_name,
            'total_configs': len(valid_results),
            'profitable_configs': len([r for r in valid_results if r['total_return'] > 0]),
            'best_pf': best_pf['profit_factor'],
            'best_pf_return': best_pf['total_return'],
            'best_return': best_return['total_return'],
            'best_return_pf': best_return['profit_factor'],
            'best_wr': best_wr['win_rate'],
            'avg_return': np.mean([r['total_return'] for r in valid_results])
        })
    
    # Print summary table
    print("\nSummary Across All Markets:")
    print("-" * 80)
    print(f"{'Market':<25} {'Configs':<10} {'Best PF':<10} {'Best Ret%':<12} {'Best WR%':<10}")
    print("-" * 80)
    
    for row in summary_rows:
        print(f"{row['pair']:<25} {row['total_configs']:<10} "
              f"{row['best_pf']:<10.2f} {row['best_return']:<12.2f} "
              f"{row['best_wr']:<10.1f}")
    
    # Save detailed results
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)
    
    for pair_name, results in all_results.items():
        valid_results = [r for r in results if r['total_trades'] >= 5]
        
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
                'regular_div': result['params']['plot_regular_bull'],
                'hidden_div': result['params']['plot_hidden_bull'],
                'total_trades': result['total_trades'],
                'winning_trades': result['winning_trades'],
                'losing_trades': result['losing_trades'],
                'win_rate': result['win_rate'],
                'total_return': result['total_return'],
                'profit_factor': result['profit_factor'],
                'max_drawdown_pct': result['max_drawdown_pct'],
                'sharpe_ratio': result['sharpe_ratio'],
                'avg_bars_held': result['avg_bars_held'],
                'final_capital': result['final_capital']
            }
            
            # Add stop loss specific params
            if result['params']['sl_type'] == 'PERC':
                row['sl_percent'] = result['params'].get('stop_loss_percent', '')
            elif result['params']['sl_type'] == 'ATR':
                row['atr_mult'] = result['params'].get('atr_multiplier', '')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by multiple criteria
        df = df.sort_values(
            ['profit_factor', 'total_return', 'max_drawdown_pct'],
            ascending=[False, False, True]
        )
        
        # Save CSV
        csv_file = output_dir / f"fast_results_{pair_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file.name} ({len(df)} configurations)")
    
    # Create combined summary
    all_data = []
    for pair_name, results in all_results.items():
        for result in results:
            if result['total_trades'] >= 5:
                result['params']['pair'] = pair_name
                all_data.append({**result['params'], **{k: v for k, v in result.items() if k != 'params'}})
    
    if all_data:
        combined_df = pd.DataFrame(all_data)
        combined_df = combined_df.sort_values(['profit_factor', 'total_return'], ascending=[False, False])
        combined_file = output_dir / "fast_results_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"✓ Saved combined results: {combined_file.name}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    
    # Print top performers across all markets
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS ACROSS ALL MARKETS")
    print("="*80)
    
    if all_data:
        top_configs = sorted(all_data, key=lambda x: x['profit_factor'], reverse=True)[:10]
        
        for idx, config in enumerate(top_configs, 1):
            print(f"\n#{idx}: {config['pair']}")
            print(f"  RSI: {config['rsi_length']} | "
                  f"Pivot: {config['pivot_left']}/{config['pivot_right']} | "
                  f"TP RSI: {config['take_profit_rsi']} | "
                  f"SL: {config['sl_type']}")
            print(f"  Divergences: Regular={config['plot_regular_bull']}, Hidden={config['plot_hidden_bull']}")
            print(f"  → Trades: {config['total_trades']} | "
                  f"Win Rate: {config['win_rate']:.1f}% | "
                  f"Return: {config['total_return']:.2f}%")
            print(f"  → Profit Factor: {config['profit_factor']:.2f} | "
                  f"Max DD: {config['max_drawdown_pct']:.2f}% | "
                  f"Sharpe: {config['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    all_results = run_fast_optimization()
    print_comprehensive_report(all_results)
    print("\n✓ Fast optimization completed!")

