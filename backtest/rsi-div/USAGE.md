# RSI Divergence Backtest - Quick Start Guide

## Directory Structure

All backtest files are now consolidated in one location:
```
/Users/foliveira/mypine/backtest/
├── market_data/          # Shared market data (180 days, multiple pairs)
│   ├── BTCUSDT_1h_180days.json
│   ├── ETHUSDT_1h_180days.json
│   └── ... (24 more files)
└── rsi-div/              # RSI Divergence strategy
    ├── venv/             # Python virtual environment
    ├── rsi_divergence_strategy.py
    ├── backtest.py
    ├── fast_backtest.py
    ├── quick_test.py
    └── README.md
```

## Quick Commands

### Run Quick Test (5 seconds)
```bash
cd /Users/foliveira/mypine/backtest/rsi-div
./venv/bin/python quick_test.py
```

### Run Fast Optimization (5-10 minutes)
```bash
cd /Users/foliveira/mypine/backtest/rsi-div
./venv/bin/python fast_backtest.py
```

### Run Full Optimization (30-60 minutes)
```bash
cd /Users/foliveira/mypine/backtest/rsi-div
./venv/bin/python backtest.py
```

## Latest Quick Test Results (BTCUSDT 1h, 180 days)

With 180 days of data, results improved significantly:

| Configuration | Return | Profit Factor | Win Rate | Drawdown |
|--------------|--------|---------------|----------|----------|
| **ATR Stop Loss** | 28.00% | 1.56 | 60.61% | 6.24% |
| **Default** | 21.93% | 1.52 | 61.63% | 10.19% |
| **5% Stop Loss** | 15.91% | 1.33 | 60.67% | 12.50% |
| **GOOGL Optimized** | 4.43% | 1.50 | 63.04% | 5.47% |

## Available Market Data

- **Pairs**: BTC, ETH, BNB, SOL, LINK, TRX, XRP (7 pairs)
- **Timeframes**: 15m, 1h, 4h, 1d (4 timeframes)
- **Period**: 180 days
- **Total**: 28 data files

## Output Files

Results are saved as CSV files in the same directory:
- `fast_results_BTCUSDT_1h_180days.csv` (per pair)
- `fast_results_combined.csv` (all results)

## What Was Changed

1. **Consolidated Structure**: Moved from two backtest folders to one at project root
2. **Updated Paths**: All scripts now point to `/Users/foliveira/mypine/backtest/`
3. **Better Data**: Using 180-day datasets instead of 90-day for more robust testing
4. **More Markets**: Added XRP to the available pairs

## Strategy Parameters Being Tested

- RSI Length: 5, 7, 9, 14
- Pivot Left/Right: Various combinations (1/3, 1/5, 2/3, 3/3)
- Take Profit RSI: 70, 75, 80, 85
- Stop Loss Types: NONE, PERC (3%, 5%), ATR (2.5x, 3.5x)
- Divergence Types: Regular only, Hidden only, Both

## Next Steps

1. **Quick validation**: Run `quick_test.py` to verify setup
2. **Fast optimization**: Run `fast_backtest.py` to find best parameters across all markets
3. **Analyze results**: Check generated CSV files for detailed metrics
4. **Apply to Pine Script**: Use best parameters in your TradingView strategy

## Tips

- Focus on configurations with Profit Factor > 1.5 and Drawdown < 10%
- Win rates > 60% are excellent for divergence strategies
- ATR-based stop losses often perform better than fixed percentage
- Results vary by market - what works for BTC may not work for altcoins
- Always validate on different time periods before live trading

