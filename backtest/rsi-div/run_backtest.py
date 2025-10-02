#!/usr/bin/env python3
"""
Unified RSI Divergence Strategy Backtesting Script
Replaces quick_test.py, fast_backtest.py, and backtest.py
Multi-threaded for faster execution
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
warnings.filterwarnings('ignore')

from rsi_divergence_strategy import RSIDivergenceStrategy
from backtest import RSIDivergenceBacktest

# Thread-safe print lock
print_lock = Lock()


def print_parameter_info():
    """Print all strategy parameters and their ranges"""
    print("\n" + "="*80)
    print("RSI DIVERGENCE STRATEGY - PARAMETER RANGES")
    print("="*80)
    
    print("\n📊 RSI SETTINGS")
    print("  • RSI Length: 5, 7, 9, 14")
    print("  • Take Profit RSI: 70, 75, 80, 85")
    print("  • RSI Source: Close price")
    
    print("\n📍 PIVOT SETTINGS")
    print("  • Pivot Left: 1, 2, 3")
    print("  • Pivot Right: 1, 3, 5")
    print("  • Lookback Range: 5-60 bars")
    
    print("\n📈 DIVERGENCE TYPES")
    print("  • Regular Bullish: ✅ Enabled (Price LL, RSI HL)")
    print("  • Hidden Bullish: Optional (Price HL, RSI LL)")
    print("  • Regular Bearish: ✅ Enabled (Price HH, RSI LH)")
    print("  • Hidden Bearish: Optional (Price LH, RSI HH)")
    
    print("\n🛑 STOP LOSS OPTIONS")
    print("  • NONE: Exit on signal or TP only")
    print("  • PERC: 3%, 5%, 7% fixed percentage")
    print("  • ATR: 2.0x, 2.5x, 3.0x, 3.5x, 4.0x ATR")
    print("  • MIN_LOW: Trailing based on lookback")
    
    print("\n⚙️  POSITION SETTINGS")
    print("  • Enable Long: ✅ Yes")
    print("  • Enable Short: ❌ No (can be enabled)")
    print("  • Position Size: 100% of capital")
    print("  • Commission: 0.045% per trade")
    
    print("\n🎯 OPTIMIZATION RANGES (Fast Mode)")
    print("  • RSI Length: [5, 7, 9, 14]")
    print("  • Pivots: [(1,3), (1,5), (2,3)]")
    print("  • TP RSI: [75, 80]")
    print("  • Stop Loss: [NONE, ATR 3.5x]")
    print("  • Divergences: [Regular only, Regular + Hidden]")
    print("  • Total Combinations: 96 per market")
    
    print("\n⚡ MULTI-THREADING")
    print("  • Workers: 4 threads")
    print("  • Speed: ~4x faster than single-threaded")
    
    print("="*80 + "\n")


def quick_test(data_file: str = None):
    """Quick test with 4 key configurations"""
    if data_file is None:
        data_file = "/Users/foliveira/mypine/backtest/market_data/BTCUSDT_1h_180days.json"
    
    # Print parameter information
    print_parameter_info()
    
    print("="*80)
    print("RSI DIVERGENCE STRATEGY - QUICK TEST")
    print("="*80)
    print(f"\nTesting with: {Path(data_file).name}")
    
    backtest = RSIDivergenceBacktest(data_file)
    if not backtest.load_data():
        return
    
    configs = [
        {
            'name': 'Default (RSI 9, Pivot 1/3, TP 80)',
            'params': {
                'rsi_length': 9, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 80, 'sl_type': 'NONE', 'enable_long': True,
                'enable_short': False, 'plot_regular_bull': True,
                'plot_hidden_bull': True, 'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        },
        {
            'name': 'With ATR Stop Loss',
            'params': {
                'rsi_length': 9, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 80, 'sl_type': 'ATR', 'atr_multiplier': 3.5,
                'enable_long': True, 'enable_short': False,
                'plot_regular_bull': True, 'plot_hidden_bull': True,
                'plot_regular_bear': True, 'plot_hidden_bear': False
            }
        },
        {
            'name': 'Fast RSI (RSI 5, TP 75)',
            'params': {
                'rsi_length': 5, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 75, 'sl_type': 'NONE', 'enable_long': True,
                'enable_short': False, 'plot_regular_bull': True,
                'plot_hidden_bull': False, 'plot_regular_bear': True,
                'plot_hidden_bear': False
            }
        },
        {
            'name': 'Conservative (1% Stop Loss)',
            'params': {
                'rsi_length': 9, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 80, 'sl_type': 'PERC', 'stop_loss_percent': 1.0,
                'enable_long': True, 'enable_short': False,
                'plot_regular_bull': True, 'plot_hidden_bull': True,
                'plot_regular_bear': True, 'plot_hidden_bear': False
            }
        },
        {
            'name': 'Conservative (2% Stop Loss)',
            'params': {
                'rsi_length': 9, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 80, 'sl_type': 'PERC', 'stop_loss_percent': 2.0,
                'enable_long': True, 'enable_short': False,
                'plot_regular_bull': True, 'plot_hidden_bull': True,
                'plot_regular_bear': True, 'plot_hidden_bear': False
            }
        },
        {
            'name': 'Conservative (5% Stop Loss)',
            'params': {
                'rsi_length': 9, 'pivot_left': 1, 'pivot_right': 3,
                'take_profit_rsi': 80, 'sl_type': 'PERC', 'stop_loss_percent': 5.0,
                'enable_long': True, 'enable_short': False,
                'plot_regular_bull': True, 'plot_hidden_bull': True,
                'plot_regular_bear': True, 'plot_hidden_bear': False
            }
        }
    ]
    
    print("\n" + "-"*80)
    print("TESTING CONFIGURATIONS")
    print("-"*80)
    
    results = []
    for config in configs:
        print(f"\n{config['name']}")
        strategy = RSIDivergenceStrategy(**config['params'])
        metrics = backtest.run_backtest(strategy)
        results.append({'name': config['name'], 'metrics': metrics})
        
        print(f"  → Trades: {metrics['total_trades']} | "
              f"Win Rate: {metrics['win_rate']:.1f}% | "
              f"Return: {metrics['total_return']:.2f}% | "
              f"PF: {metrics['profit_factor']:.2f} | "
              f"DD: {metrics['max_drawdown_pct']:.2f}%")
    
    print("\n" + "="*80)
    print("RANKED BY TOTAL RETURN")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['metrics']['total_return'], reverse=True)
    for idx, result in enumerate(results_sorted, 1):
        m = result['metrics']
        print(f"\n#{idx}: {result['name']}")
        print(f"  Return: {m['total_return']:.2f}% | PF: {m['profit_factor']:.2f} | "
              f"Win Rate: {m['win_rate']:.1f}% | Trades: {m['total_trades']}")


def run_single_backtest(args):
    """Run a single backtest (for threading)"""
    config, data_file = args
    backtest = RSIDivergenceBacktest(str(data_file))
    if not backtest.load_data():
        return None
    
    strategy = RSIDivergenceStrategy(**config)
    metrics = backtest.run_backtest(strategy)
    return metrics


def generate_test_configs(extensive: bool = False):
    """Generate parameter combinations to test"""
    if extensive:
        # EXTENSIVE mode - test many more combinations
        rsi_lengths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        pivot_combos = [(1, 1), (1, 3), (1, 5), (2, 2), (2, 3), (3, 3), (3, 5)]
        tp_rsi_levels = [70, 75, 80, 85]
        
        # All stop loss configurations
        sl_configs = [
            {'sl_type': 'NONE'},
            {'sl_type': 'PERC', 'stop_loss_percent': 3.0},
            {'sl_type': 'PERC', 'stop_loss_percent': 5.0},
            {'sl_type': 'PERC', 'stop_loss_percent': 7.0},
            {'sl_type': 'ATR', 'atr_multiplier': 2.0},
            {'sl_type': 'ATR', 'atr_multiplier': 2.5},
            {'sl_type': 'ATR', 'atr_multiplier': 3.0},
            {'sl_type': 'ATR', 'atr_multiplier': 3.5},
            {'sl_type': 'ATR', 'atr_multiplier': 4.0},
            {'sl_type': 'MIN_LOW', 'min_low_lookback': 10},
            {'sl_type': 'MIN_LOW', 'min_low_lookback': 15},
        ]
        
        div_configs = [
            {'regular': True, 'hidden': False},   # Regular only
            {'regular': False, 'hidden': True},   # Hidden only
            {'regular': True, 'hidden': True}     # Both
        ]
    else:
        # FAST mode - curated best performers
        rsi_lengths = [5, 7, 9, 14]
        pivot_combos = [(1, 3), (1, 5), (2, 3)]
        tp_rsi_levels = [75, 80]
        sl_configs = [
            {'sl_type': 'NONE'},
            {'sl_type': 'ATR', 'atr_multiplier': 3.5}
        ]
        div_configs = [
            {'regular': True, 'hidden': False},
            {'regular': True, 'hidden': True}
        ]
    
    # Generate all combinations
    test_configs = []
    for rsi_len in rsi_lengths:
        for pivot_combo in pivot_combos:
            for tp_rsi in tp_rsi_levels:
                for sl_config in sl_configs:
                    for divs in div_configs:
                        config = {
                            'rsi_length': rsi_len,
                            'pivot_left': pivot_combo[0],
                            'pivot_right': pivot_combo[1],
                            'take_profit_rsi': tp_rsi,
                            'enable_long': True,
                            'enable_short': False,
                            'plot_regular_bull': divs['regular'],
                            'plot_hidden_bull': divs['hidden'],
                            'plot_regular_bear': True,
                            'plot_hidden_bear': False,
                            **sl_config
                        }
                        test_configs.append(config)
    
    return test_configs


def fast_optimization(output_dir: Path = None, num_workers: int = 4, extensive: bool = False, data_file: str = None):
    """Fast optimization with curated parameters and multi-threading"""
    if output_dir is None:
        output_dir = Path("/Users/foliveira/mypine/backtest/rsi-div")
    
    # Print parameter information
    print_parameter_info()
    
    # Select data files
    if data_file:
        # Single file specified
        data_files = [Path(data_file)]
    else:
        # All 1h files
        data_dir = Path("/Users/foliveira/mypine/backtest/market_data")
        data_files = sorted(data_dir.glob("*_1h_180days.json"))
    
    mode_name = "EXTENSIVE" if extensive else "FAST"
    print("="*80)
    print(f"RSI DIVERGENCE - {mode_name} OPTIMIZATION (MULTI-THREADED)")
    print("="*80)
    
    # Generate parameter combinations
    test_configs = generate_test_configs(extensive=extensive)
    
    print(f"\nConfigurations to test: {len(test_configs)}")
    print(f"Market files: {len(data_files)}")
    print(f"Total tests: {len(test_configs) * len(data_files)}")
    print(f"Using {num_workers} worker threads")
    print(f"Estimated time: {len(test_configs) * len(data_files) / num_workers * 0.05:.1f} seconds\n")
    
    start_time = time.time()
    all_results = {}
    
    for data_file in data_files:
        pair_name = data_file.stem
        print(f"\nTesting {pair_name}...")
        
        # Prepare work items (config, data_file) tuples
        work_items = [(config, data_file) for config in test_configs]
        
        results = []
        completed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_config = {executor.submit(run_single_backtest, item): item for item in work_items}
            
            # Process completed tasks
            for future in as_completed(future_to_config):
                completed += 1
                
                if completed % 20 == 0 or completed == len(work_items):
                    progress_pct = (completed / len(work_items)) * 100
                    with print_lock:
                        print(f"  Progress: {completed}/{len(work_items)} ({progress_pct:.1f}%)")
                
                try:
                    metrics = future.result()
                    if metrics:
                        results.append(metrics)
                except Exception as e:
                    with print_lock:
                        print(f"  ⚠️  Error in backtest: {e}")
        
        all_results[pair_name] = results
        
        # Show top 3
        valid = [r for r in results if r['total_trades'] >= 5 and r['total_return'] > 0]
        if valid:
            top = sorted(valid, key=lambda x: (x['profit_factor'], -x['max_drawdown_pct']), reverse=True)[:3]
            print(f"\n  Top 3 for {pair_name}:")
            for idx, r in enumerate(top, 1):
                p = r['params']
                print(f"    #{idx}: RSI{p['rsi_length']} P{p['pivot_left']}/{p['pivot_right']} "
                      f"TP{p['take_profit_rsi']} {p['sl_type']} → "
                      f"{r['total_return']:.1f}% (PF:{r['profit_factor']:.2f})")
    
    elapsed_time = time.time() - start_time
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    for pair_name, results in all_results.items():
        valid = [r for r in results if r['total_trades'] >= 5]
        if not valid:
            continue
        
        rows = []
        for r in valid:
            row = {
                'pair': pair_name,
                'rsi_length': r['params']['rsi_length'],
                'pivot_lr': f"{r['params']['pivot_left']}/{r['params']['pivot_right']}",
                'tp_rsi': r['params']['take_profit_rsi'],
                'sl_type': r['params']['sl_type'],
                'trades': r['total_trades'],
                'win_rate': r['win_rate'],
                'return_pct': r['total_return'],
                'profit_factor': r['profit_factor'],
                'max_dd_pct': r['max_drawdown_pct'],
                'sharpe': r['sharpe_ratio']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['profit_factor', 'return_pct'], ascending=[False, False])
        
        csv_file = output_dir / f"results_{pair_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ {csv_file.name}")
    
    print(f"\n✓ Optimization complete! Check {output_dir}/ for CSV results.")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"⚡ Speed: {len(test_configs) * len(data_files) / elapsed_time:.1f} backtests/second")


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description='RSI Divergence Strategy Backtesting (Multi-threaded)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                           # Quick test (5 seconds, 4 configs)
  %(prog)s --quick --file ETHUSDT_1h.json    # Quick test on specific pair
  %(prog)s --optimize                        # Fast optimization (2-3 min, 96 configs)
  %(prog)s --optimize --extensive            # Extensive optimization (10-15 min, 9,240 configs!)
  %(prog)s --optimize --workers 8            # Use 8 threads for faster execution
  %(prog)s --optimize --extensive --workers 8  # Full power mode!

Modes:
  --quick              Test 4 pre-selected configurations (fastest validation)
  --optimize           Test 96 curated combinations (good balance)
  --optimize --extensive   Test 9,240 combinations (exhaustive search)
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with 4 key configurations')
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization across all markets')
    parser.add_argument('--extensive', action='store_true',
                       help='Use extensive parameter grid (9,240 combinations vs 96)')
    parser.add_argument('--file', type=str,
                       help='Specific market data file to test')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test(args.file)
    elif args.optimize:
        output_dir = Path(args.output) if args.output else None
        fast_optimization(output_dir, num_workers=args.workers, extensive=args.extensive, data_file=args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

