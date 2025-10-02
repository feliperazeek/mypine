#!/usr/bin/env python3
"""
RSI Divergence Strategy Implementation
Converted from Pine Script to Python
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


class RSIDivergenceStrategy:
    """
    RSI Divergence Trading Strategy
    
    Detects regular and hidden divergences between price and RSI to identify
    potential trend reversals and continuations.
    """
    
    def __init__(
        self,
        rsi_length: int = 9,
        pivot_left: int = 1,
        pivot_right: int = 3,
        range_upper: int = 60,
        range_lower: int = 5,
        take_profit_rsi: int = 80,
        sl_type: str = "NONE",
        stop_loss_percent: float = 5.0,
        atr_length: int = 14,
        atr_multiplier: float = 3.5,
        min_low_lookback: int = 12,
        enable_long: bool = True,
        enable_short: bool = False,
        plot_regular_bull: bool = True,
        plot_hidden_bull: bool = True,
        plot_regular_bear: bool = True,
        plot_hidden_bear: bool = False
    ):
        """Initialize strategy with parameters"""
        self.rsi_length = rsi_length
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.range_upper = range_upper
        self.range_lower = range_lower
        self.take_profit_rsi = take_profit_rsi
        self.sl_type = sl_type
        self.stop_loss_percent = stop_loss_percent
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.min_low_lookback = min_low_lookback
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.plot_regular_bull = plot_regular_bull
        self.plot_hidden_bull = plot_hidden_bull
        self.plot_regular_bear = plot_regular_bear
        self.plot_hidden_bear = plot_hidden_bear
        
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def find_pivot_low(self, series: pd.Series, left: int, right: int) -> pd.Series:
        """
        Find pivot lows in a series
        A pivot low is a value that is lower than 'left' values before it
        and 'right' values after it
        """
        pivot_lows = pd.Series(index=series.index, dtype=float)
        
        for i in range(left, len(series) - right):
            window_start = i - left
            window_end = i + right + 1
            window = series.iloc[window_start:window_end]
            
            if series.iloc[i] == window.min():
                pivot_lows.iloc[i] = series.iloc[i]
        
        return pivot_lows
    
    def find_pivot_high(self, series: pd.Series, left: int, right: int) -> pd.Series:
        """
        Find pivot highs in a series
        A pivot high is a value that is higher than 'left' values before it
        and 'right' values after it
        """
        pivot_highs = pd.Series(index=series.index, dtype=float)
        
        for i in range(left, len(series) - right):
            window_start = i - left
            window_end = i + right + 1
            window = series.iloc[window_start:window_end]
            
            if series.iloc[i] == window.max():
                pivot_highs.iloc[i] = series.iloc[i]
        
        return pivot_highs
    
    def in_range(self, bars_since: int) -> bool:
        """Check if bars since last pivot is within valid range"""
        return self.range_lower <= bars_since <= self.range_upper
    
    def detect_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all types of divergences in the data
        Returns dataframe with divergence signals
        """
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_length)
        
        # Calculate ATR for stop loss
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_length)
        
        # Find pivot points in RSI
        rsi_pivot_lows = self.find_pivot_low(df['rsi'], self.pivot_left, self.pivot_right)
        rsi_pivot_highs = self.find_pivot_high(df['rsi'], self.pivot_left, self.pivot_right)
        
        # Initialize signal columns
        df['bullish_div'] = False
        df['hidden_bullish_div'] = False
        df['bearish_div'] = False
        df['hidden_bearish_div'] = False
        df['long_signal'] = False
        df['short_signal'] = False
        
        # Track last pivot indices
        last_rsi_low_idx = None
        last_price_low = None
        last_rsi_low = None
        
        last_rsi_high_idx = None
        last_price_high = None
        last_rsi_high = None
        
        # Scan for divergences
        for i in range(self.pivot_left + self.pivot_right, len(df)):
            current_idx = i - self.pivot_right
            
            # Check for RSI pivot low at current_idx
            if not pd.isna(rsi_pivot_lows.iloc[current_idx]):
                if last_rsi_low_idx is not None:
                    bars_since = current_idx - last_rsi_low_idx
                    
                    if self.in_range(bars_since):
                        current_price_low = df['low'].iloc[current_idx]
                        current_rsi_low = df['rsi'].iloc[current_idx]
                        
                        # Regular Bullish Divergence: Price LL, RSI HL
                        if (self.plot_regular_bull and 
                            current_price_low < last_price_low and 
                            current_rsi_low > last_rsi_low):
                            df.loc[df.index[i], 'bullish_div'] = True
                            if self.enable_long:
                                df.loc[df.index[i], 'long_signal'] = True
                        
                        # Hidden Bullish Divergence: Price HL, RSI LL
                        if (self.plot_hidden_bull and 
                            current_price_low > last_price_low and 
                            current_rsi_low < last_rsi_low):
                            df.loc[df.index[i], 'hidden_bullish_div'] = True
                            if self.enable_long:
                                df.loc[df.index[i], 'long_signal'] = True
                
                # Update last pivot low
                last_rsi_low_idx = current_idx
                last_price_low = df['low'].iloc[current_idx]
                last_rsi_low = df['rsi'].iloc[current_idx]
            
            # Check for RSI pivot high at current_idx
            if not pd.isna(rsi_pivot_highs.iloc[current_idx]):
                if last_rsi_high_idx is not None:
                    bars_since = current_idx - last_rsi_high_idx
                    
                    if self.in_range(bars_since):
                        current_price_high = df['high'].iloc[current_idx]
                        current_rsi_high = df['rsi'].iloc[current_idx]
                        
                        # Regular Bearish Divergence: Price HH, RSI LH
                        if (self.plot_regular_bear and 
                            current_price_high > last_price_high and 
                            current_rsi_high < last_rsi_high):
                            df.loc[df.index[i], 'bearish_div'] = True
                            if self.enable_short:
                                df.loc[df.index[i], 'short_signal'] = True
                        
                        # Hidden Bearish Divergence: Price LH, RSI HH
                        if (self.plot_hidden_bear and 
                            current_price_high < last_price_high and 
                            current_rsi_high > last_rsi_high):
                            df.loc[df.index[i], 'hidden_bearish_div'] = True
                            if self.enable_short:
                                df.loc[df.index[i], 'short_signal'] = True
                
                # Update last pivot high
                last_rsi_high_idx = current_idx
                last_price_high = df['high'].iloc[current_idx]
                last_rsi_high = df['rsi'].iloc[current_idx]
        
        return df
    
    def calculate_stop_loss(self, df: pd.DataFrame, idx: int, position_type: str) -> float:
        """Calculate stop loss price based on sl_type"""
        current_price = df['close'].iloc[idx]
        
        if self.sl_type == "PERC":
            if position_type == "long":
                return current_price * (1 - self.stop_loss_percent / 100)
            else:  # short
                return current_price * (1 + self.stop_loss_percent / 100)
        
        elif self.sl_type == "ATR":
            atr_value = df['atr'].iloc[idx]
            if position_type == "long":
                return current_price - (atr_value * self.atr_multiplier)
            else:  # short
                return current_price + (atr_value * self.atr_multiplier)
        
        elif self.sl_type == "MIN_LOW":
            if position_type == "long":
                lookback_start = max(0, idx - self.min_low_lookback)
                return df['low'].iloc[lookback_start:idx+1].min()
            else:  # short
                lookback_start = max(0, idx - self.min_low_lookback)
                return df['high'].iloc[lookback_start:idx+1].max()
        
        return None
    
    def get_strategy_params(self) -> Dict:
        """Return strategy parameters as dictionary"""
        return {
            'rsi_length': self.rsi_length,
            'pivot_left': self.pivot_left,
            'pivot_right': self.pivot_right,
            'range_upper': self.range_upper,
            'range_lower': self.range_lower,
            'take_profit_rsi': self.take_profit_rsi,
            'sl_type': self.sl_type,
            'stop_loss_percent': self.stop_loss_percent,
            'atr_length': self.atr_length,
            'atr_multiplier': self.atr_multiplier,
            'min_low_lookback': self.min_low_lookback,
            'enable_long': self.enable_long,
            'enable_short': self.enable_short,
            'plot_regular_bull': self.plot_regular_bull,
            'plot_hidden_bull': self.plot_hidden_bull,
            'plot_regular_bear': self.plot_regular_bear,
            'plot_hidden_bear': self.plot_hidden_bear
        }

