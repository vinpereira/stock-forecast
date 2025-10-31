# Stock Forecast

Professional stock price forecasting with Facebook Prophet.

## Features

- 📊 **Prophet Time Series Forecasting**
- 🎯 **Optimal Sell Date Recommendation**
- 📈 **Multiple Scenarios** (Optimistic/Expected/Pessimistic)
- 📉 **Volatility Analysis**
- 🎨 **Beautiful Visualizations**
- 📁 **CSV Export with Components**
- 🔧 **Highly Configurable**

## Setup

### Prerequisites
- Python 3.10+
- uv (recommended) or pip

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd stock-forecast

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

### Basic Usage
```bash
uv run python main.py
```

### With Custom Parameters
```bash
# Custom stock symbol
uv run python main.py --symbol GOOGL

# Custom forecast period
uv run python main.py --days 30

# Both
uv run python main.py --symbol MSFT --days 60
```

## Configuration

Edit `config.yaml` to customize:
```yaml
# Stock Configuration
stock:
  symbol: "AAPL"
  start: "2023-01-01"
  end: "today"

# Forecast Settings
forecast:
  days: 365
  freq: "D"

# Prophet Model Parameters
model:
  changepoint_prior_scale: 0.05
  seasonality_prior_scale: 10
  holidays_prior_scale: 15
  weekly_seasonality: true
  yearly_seasonality: true
  daily_seasonality: false
  country_holidays: "US"

# Output Settings
output:
  directory: "./outputs"
  plot_dpi: 300
  save_csv: true
```

## Output

The forecasting pipeline generates:

### 1. Forecast Plot
- Historical data (black dots)
- Expected forecast (blue line)
- Optimistic scenario (green dashed)
- Pessimistic scenario (red dashed)
- 95% confidence interval (gray area)
- Value annotations at forecast end

### 2. Components Plot
- Trend decomposition
- Yearly seasonality
- Weekly seasonality
- Holiday effects

### 3. CSV Export
- Complete forecast data
- All Prophet components
- Confidence intervals

### 4. Console Summary
- Current price
- 30-day and 90-day forecasts
- Optimal sell date
- Volatility metrics

## Project Structure
```
stock-forecast/
├── src/
│   ├── analysis/          # Forecast analysis
│   │   └── forecast.py    # ForecastAnalyzer class
│   ├── data/              # Data fetching and preprocessing
│   │   ├── fetcher.py     # Yahoo Finance fetcher
│   │   └── preprocessor.py # Data preparation
│   ├── models/            # Prophet model wrapper
│   │   └── prophet_model.py
│   ├── utils/             # Configuration utilities
│   │   └── config.py
│   └── visualization/     # Plotting
│       └── plotter.py     # ForecastPlotter class
├── tests/                 # Test suite (coming soon)
├── outputs/               # Generated outputs
├── config.yaml            # Configuration file
├── main.py               # Main entry point
├── pyproject.toml        # Project metadata
└── README.md             # This file
```

## Example Output
```
================================================================================
📊 STOCK PRICE FORECASTING WITH PROPHET
================================================================================

🔧 Step 1: Loading Configuration...
   Symbol: AAPL
   Period: 2023-01-01 to 2025-10-31
   Forecast: 365 days

📥 Step 2: Fetching Data for AAPL...
   ✓ Fetched 706 records

🧹 Step 3: Preprocessing Data...
   Price range: $120.45 - $237.23
   Mean price: $178.56

🤖 Step 4: Training Prophet Model...
   ✓ Trained on 706 samples

🔮 Step 5: Generating Forecasts...
   ✓ Forecast generated!

📊 Step 6: Creating Visualizations...
   ✓ Forecast plot saved
   ✓ Components plot saved

📈 Step 7: Analyzing Results...

================================================================================
📊 FORECAST SUMMARY - AAPL
================================================================================

💰 Current Price: $237.23

📈 30-Day Forecast:
   Expected:    $245.67
   Optimistic:  $268.91
   Pessimistic: $222.43
   Change: +3.56%

📈 90-Day Forecast:
   Expected:    $259.12
   Optimistic:  $295.78
   Pessimistic: $222.46
   Change: +9.23%

🎯 Optimal Sell Date: 2026-10-29
   Expected Price: $301.97
   Days from now: 364

📊 Volatility (90-day):
   Std Dev: $12.34
   Avg Confidence Range: $73.25

================================================================================

✅ FORECASTING COMPLETED SUCCESSFULLY!
```

## Development

### Project Status
- [x] Project setup and basic structure
- [x] Analysis module and main pipeline
- [ ] Enhancements and robustness
- [ ] Testing and documentation

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Facebook Prophet for the forecasting model
- Yahoo Finance for stock data
- The Python data science community
