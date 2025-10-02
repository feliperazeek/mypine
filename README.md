# mypine
Collection of Pine Script trading strategies and backtesting tools.

## Overview
This repository contains various Pine Script strategies focused on:
- EMA-based crossover strategies
- Pullback and reversal patterns  
- RSI divergence detection
- Hotzone analysis
- Backtesting utilities

## Structure
- `nakinvest/` - Main strategy files (.pine scripts for TradingView)
  - `indicators/` - Custom Pine Script indicators
  - `module2/` - Utility modules for candle pattern identification
- `backtest/` - Python backtesting tools and market data
  - `market_data/` - Historical price data (180 days, 7 pairs, 4 timeframes)
  - `rsi-div/` - RSI Divergence strategy Python implementation with optimization

## Key Strategies

### Pine Script (TradingView)
- **cross-ema.pine** - EMA crossover strategy
- **cross-ema-rsi.pine** - EMA crossovers with RSI confirmation
- **ema20-pullback.pine** - EMA20 pullback strategy
- **hotzone-pullback.pine** - Pullback strategy with hotzone analysis
- **three-bar-reversal.pine** - Three-bar reversal pattern detection
- **rsi-divergences.pine** - Professional RSI divergence strategy

### Python Backtesting
- **RSI Divergence Strategy** - Converted from Pine Script with parameter optimization
  - Regular & Hidden divergences (Bullish/Bearish)
  - Multiple stop loss types (Percentage, ATR, Trailing)
  - Comprehensive backtesting framework
  - Parameter optimization across all markets

## Quick Start

### Pine Script Strategies
Open any `.pine` file in TradingView Pine Editor and run on your charts.

### Python Backtesting (RSI Divergence)
```bash
cd backtest/rsi-div
./venv/bin/python quick_test.py  # Quick test (5 seconds)
./venv/bin/python fast_backtest.py  # Full optimization (5-10 min)
```

See `backtest/rsi-div/USAGE.md` for detailed instructions.

## Market Data
Includes historical data for 7 crypto pairs:
- **Pairs**: BTC, ETH, BNB, SOL, LINK, TRX, XRP
- **Timeframes**: 15m, 1h, 4h, 1d
- **Period**: 180 days
- **Total**: 28 data files (JSON format)

## Latest Backtest Results

RSI Divergence Strategy (BTCUSDT 1h, 180 days):
- **Best Configuration**: ATR Stop Loss - 28% return, 1.56 PF, 60.61% win rate
- **Default Settings**: 21.93% return, 1.52 PF, 61.63% win rate
- All configurations tested across 7 pairs and 4 timeframes

## Features
- ✅ Professional Pine Script strategies for TradingView
- ✅ Python implementation with full backtesting framework
- ✅ Parameter optimization with comprehensive metrics
- ✅ 180 days of historical data across multiple markets
- ✅ CSV export of all results for analysis
- ✅ Profit factor, win rate, drawdown, Sharpe ratio metrics
