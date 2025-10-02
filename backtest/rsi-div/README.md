# RSI Divergence Strategy Backtesting

Python implementation of the RSI Divergence trading strategy, converted from Pine Script.

## Overview

This strategy detects divergences between price action and RSI indicator to identify potential trend reversals and continuations. It supports:

- **Regular Bullish Divergence**: Price makes lower low, RSI makes higher low (reversal signal)
- **Hidden Bullish Divergence**: Price makes higher low, RSI makes lower low (continuation signal)
- **Regular Bearish Divergence**: Price makes higher high, RSI makes lower high (reversal signal)
- **Hidden Bearish Divergence**: Price makes lower high, RSI makes higher high (continuation signal)

## Files

- `rsi_divergence_strategy.py` - Core strategy implementation
- `backtest.py` - Comprehensive backtesting with parameter optimization
- `quick_test.py` - Quick test with a few configurations
- `fast_backtest.py` - Faster optimization with reduced parameter space

## Installation

1. Create virtual environment:
```bash
cd /Users/foliveira/mypine/backtest/rsi-div
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install pandas numpy
```

## Usage

### Quick Test (Recommended for first run)

Test a few configurations on one market:
```bash
source venv/bin/activate
python quick_test.py
```

### Fast Optimization

Test optimized parameter combinations across all markets:
```bash
source venv/bin/activate
python fast_backtest.py
```

This will:
- Test ~100-200 parameter combinations per market
- Complete in 5-10 minutes
- Generate CSV reports sorted by profit factor and drawdown

### Full Optimization (Takes longer)

Test extensive parameter combinations:
```bash
source venv/bin/activate
python backtest.py
```

This will:
- Test 1000+ parameter combinations per market
- Take 30-60 minutes depending on CPU
- Generate comprehensive CSV reports

## Strategy Parameters

### RSI Settings
- `rsi_length` (5-14): Period for RSI calculation
- `take_profit_rsi` (70-85): RSI level to take profit

### Pivot Settings
- `pivot_left` (1-3): Lookback bars to the left
- `pivot_right` (1-5): Lookback bars to the right

### Divergence Settings
- `range_upper` (60): Maximum bars between pivots
- `range_lower` (5): Minimum bars between pivots
- `plot_regular_bull`: Enable regular bullish divergence
- `plot_hidden_bull`: Enable hidden bullish divergence
- `plot_regular_bear`: Enable regular bearish divergence
- `plot_hidden_bear`: Enable hidden bearish divergence

### Risk Management
- `sl_type`: Stop loss type (NONE, PERC, ATR, MIN_LOW)
- `stop_loss_percent` (3-7): Percentage stop loss
- `atr_multiplier` (2-4): ATR-based stop loss multiplier
- `atr_length` (14): ATR calculation period
- `min_low_lookback` (12): Lookback for trailing stop

### Position Settings
- `enable_long`: Enable long positions
- `enable_short`: Enable short positions

## Results

Results are saved as CSV files in the same directory:
- `results_BTCUSDT_1h_90days.csv`
- `results_ETHUSDT_1h_90days.csv`
- etc.

Each CSV contains:
- All parameter combinations tested
- Performance metrics (win rate, profit factor, drawdown, etc.)
- Sorted by profit factor and drawdown

## Performance Metrics

- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of winning trades
- **Total Return**: Percentage return on initial capital
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return
- **Avg Bars Held**: Average holding period

## Example Results (BTCUSDT 1h 180 days data)

From quick test:
1. **With ATR Stop Loss**: ~10% return, PF: 1.5+, 60%+ win rate
2. **Default Settings**: ~7% return, PF: 1.5+, 63%+ win rate
3. **GOOGL Optimized**: ~6% return, PF: 3.9+, 73%+ win rate

Results will vary based on market conditions and data period.

## Notes

- Strategy performance varies significantly by market and timeframe
- Past performance does not guarantee future results
- Consider transaction costs (0.045% commission is included)
- Backtests use 100% of capital per trade (adjust in production)
- No pyramiding (one position at a time)

## Optimization Notes

Based on Pine Script comments, optimal settings for different instruments:
- **GOOGL**: RSI 5, Pivot 1/3, TP RSI 75 = 87.21% win rate
- **SPY**: RSI 5, Pivot 3/3, TP RSI 70 = 80.34% win rate

These are starting points - always validate with your specific market data.

