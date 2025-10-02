# Backtest Directory

This directory contains all backtesting tools, scripts, and market data for the mypine project.

## Structure

```
backtest/
├── market_data/              # Historical price data (180 days)
│   ├── BTCUSDT_1h_180days.json
│   ├── ETHUSDT_1h_180days.json
│   └── ... (28 files total)
│
├── rsi-div/                  # RSI Divergence Strategy
│   ├── rsi_divergence_strategy.py
│   ├── backtest.py
│   ├── fast_backtest.py
│   ├── quick_test.py
│   ├── venv/
│   ├── README.md
│   ├── USAGE.md
│   └── SUMMARY.md
│
├── strategy_backtest.py      # EMA/Hotzone strategy comparison
├── real_data_optimizer.py    # Strategy optimizer utility
├── data_fetcher.py           # Binance data fetcher
└── download_all_data.py      # Bulk market data downloader
```

## Market Data

### Available Pairs
- **BTC/USDT** - Bitcoin
- **ETH/USDT** - Ethereum  
- **BNB/USDT** - Binance Coin
- **SOL/USDT** - Solana
- **LINK/USDT** - Chainlink
- **TRX/USDT** - Tron
- **XRP/USDT** - Ripple

### Timeframes
- **15m** - 15 minutes
- **1h** - 1 hour
- **4h** - 4 hours
- **1d** - Daily

### Data Format
JSON files with OHLCV data:
```json
[
  {
    "timestamp": 1234567890000,
    "open": 50000.0,
    "high": 51000.0,
    "low": 49500.0,
    "close": 50500.0,
    "volume": 1234.56
  },
  ...
]
```

## Scripts

### RSI Divergence Strategy (`rsi-div/`)
Complete Python implementation of the RSI Divergence trading strategy with parameter optimization.

**Quick Start:**
```bash
cd rsi-div
./venv/bin/python quick_test.py       # Test in 5 seconds
./venv/bin/python fast_backtest.py    # Optimize in 5-10 min
```

See `rsi-div/USAGE.md` for detailed instructions.

### Strategy Comparison (`strategy_backtest.py`)
Compares EMA20 Pullback vs Hotzone Pullback strategies across market data.

**Usage:**
```bash
python3 strategy_backtest.py
```

### Data Tools

#### Download All Data (`download_all_data.py`)
Downloads historical data for all pairs and timeframes from Binance.

**Usage:**
```bash
python3 download_all_data.py
```

#### Data Fetcher (`data_fetcher.py`)
Utility module for fetching data from Binance API.

#### Real Data Optimizer (`real_data_optimizer.py`)
General purpose strategy optimizer for testing parameter combinations.

## Requirements

### Python Packages
```bash
pip install pandas numpy requests
```

Or use the virtual environment in `rsi-div/venv/`:
```bash
cd rsi-div
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

## Performance Notes

- **RSI Divergence**: Best performing with ATR stop loss (28% return, 1.56 PF on BTCUSDT 1h)
- **EMA Pullback**: Solid win rates on trending markets
- **Hotzone Pullback**: Good for identifying key support/resistance levels

## Data Updates

To refresh market data:
```bash
python3 download_all_data.py
```

This will download the latest 180 days of data for all configured pairs and timeframes.

## Output Files

Backtest results are saved as CSV files:
- `rsi-div/fast_results_*.csv` - Per-market results
- `rsi-div/fast_results_combined.csv` - All results combined

## Important Notes

⚠️ **Trading Risk**
- Past performance does not guarantee future results
- Always validate strategies on recent data
- Use proper risk management in live trading

⚠️ **Commission Costs**
- Backtests include 0.045% commission per trade
- Adjust based on your broker/exchange fees

⚠️ **Position Sizing**
- Backtests use 100% of capital per trade
- Use 1-2% per trade in live trading

## Next Steps

1. Run RSI divergence optimization: `cd rsi-div && ./venv/bin/python fast_backtest.py`
2. Analyze results in generated CSV files
3. Apply best parameters to Pine Script strategies
4. Forward test on recent data before live trading

For questions or issues, see individual README files in each directory.

