"""
RSI Divergence Backtesting Module
Contains the core backtesting engine
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict


class RSIDivergenceBacktest:
    """Backtest engine for RSI Divergence Strategy"""
    
    def __init__(self, data_file: str, initial_capital: float = 10000):
        """Initialize backtest"""
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.data = None
        
    def load_data(self) -> bool:
        """Load market data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
            
            df = pd.DataFrame(raw_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            self.data = df
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def run_backtest(self, strategy) -> Dict:
        """
        Run backtest with given strategy
        Returns performance metrics
        """
        # Detect divergences and generate signals
        df = strategy.detect_divergences(self.data.copy())
        
        # Initialize tracking
        position = None
        entry_price = 0
        entry_idx = 0
        stop_loss = None
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        position_size = 0
        
        commission_rate = 0.00045  # 0.045%
        
        # Simulate trading
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # Check exit conditions if in position
            if position == 'long':
                # Check stop loss
                if stop_loss is not None and current_row['low'] <= stop_loss:
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
                
                # Check take profit
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
                
                # Check for opposite signal
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
                
                # Update trailing stop
                elif strategy.sl_type == "MIN_LOW" and stop_loss is not None:
                    lookback_start = max(0, i - strategy.min_low_lookback)
                    new_stop = df['low'].iloc[lookback_start:i+1].min()
                    stop_loss = max(stop_loss, new_stop)
            
            elif position == 'short':
                # Check stop loss
                if stop_loss is not None and current_row['high'] >= stop_loss:
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
                
                # Check take profit
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
                
                # Check for opposite signal
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
                
                # Update trailing stop
                elif strategy.sl_type == "MIN_LOW" and stop_loss is not None:
                    lookback_start = max(0, i - strategy.min_low_lookback)
                    new_stop = df['high'].iloc[lookback_start:i+1].max()
                    stop_loss = min(stop_loss, new_stop)
            
            # Check entry conditions if not in position
            if position is None:
                # Long entry
                if current_row['long_signal']:
                    entry_price = current_row['close']
                    position_size = current_capital * 1.0 / entry_price
                    entry_idx = i
                    position = 'long'
                    stop_loss = strategy.calculate_stop_loss(df, i, 'long')
                
                # Short entry
                elif current_row['short_signal']:
                    entry_price = current_row['close']
                    position_size = current_capital * 1.0 / entry_price
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
        metrics = self._calculate_metrics(trades, equity_curve, strategy.get_strategy_params())
        return metrics
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], params: Dict) -> Dict:
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
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown
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
        
        # Sharpe ratio
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

