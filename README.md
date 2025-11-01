# Stock Forecast

Professional stock price forecasting with Facebook Prophet.

## Features

- 📊 **Prophet Time Series Forecasting** - Advanced ML forecasting model
- 🎯 **Optimal Sell Date** - Find the best date to sell based on predictions
- 📈 **Multiple Scenarios** - Optimistic, Expected, and Pessimistic forecasts
- 📉 **Volatility Analysis** - Understand forecast uncertainty
- 🎨 **Beautiful Visualizations** - Interactive plots with annotations
- 📁 **CSV Export** - Export forecasts with all components
- 🔧 **Highly Configurable** - Easy YAML configuration
- 📓 **Jupyter Notebooks** - Interactive examples for experimentation

## Quick Start
```bash
# Install dependencies
make install

# Run forecast
make run

# Run tests
make test
```

## Installation

### Prerequisites
- Python 3.10+
- uv (recommended) or pip

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd stock-forecast

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Command Line
```bash
# Basic usage (uses config.yaml)
uv run python main.py

# Custom stock symbol
uv run python main.py --symbol GOOGL

# Custom forecast period
uv run python main.py --days 30

# Both
uv run python main.py --symbol MSFT --days 60
```

### Makefile Commands
```bash
make install    # Install dependencies
make run        # Run forecast
make test       # Run tests
make clean      # Clean outputs
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/
# - 01_explore_data.ipynb - Data exploration
# - 02_forecast_analysis.ipynb - Interactive forecasting
```

## Configuration

Edit `config.yaml` to customize behavior:
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

### Console Summary
```
================================================================================
📊 FORECAST SUMMARY - AAPL
================================================================================

💰 Current Price: $237.23

📈 30-Day Forecast:
   Expected:    $245.67
   Optimistic:  $268.91
   Pessimistic: $222.43
   Change: +3.56%

🎯 Optimal Sell Date: 2026-10-29
   Expected Price: $301.97
   Days from now: 364

📊 Volatility (90-day):
   Std Dev: $12.34
   Avg Confidence Range: $73.25
```

### Generated Files

1. **forecast_{SYMBOL}.png** - Main forecast visualization
   - Historical data points
   - Expected forecast line
   - Optimistic/pessimistic scenarios
   - 95% confidence interval
   - Value annotations

2. **forecast_components_{SYMBOL}.png** - Component decomposition
   - Trend
   - Yearly seasonality
   - Weekly seasonality
   - Holiday effects

3. **forecast_{SYMBOL}.csv** - Complete forecast data
   - All dates and predictions
   - Prophet components
   - Confidence intervals

## Project Structure
```
stock-forecast/
├── src/
│   ├── analysis/          # Forecast analysis
│   │   └── forecast.py
│   ├── data/              # Data fetching and preprocessing
│   │   ├── fetcher.py
│   │   └── preprocessor.py
│   ├── models/            # Prophet model wrapper
│   │   └── prophet_model.py
│   ├── utils/             # Configuration
│   │   └── config.py
│   └── visualization/     # Plotting
│       └── plotter.py
├── tests/                 # Test suite
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_analysis.py
│   └── test_integration.py
├── notebooks/             # Jupyter notebooks
│   ├── 01_explore_data.ipynb
│   └── 02_forecast_analysis.ipynb
├── outputs/               # Generated outputs
├── config.yaml            # Configuration
├── main.py               # Entry point
├── Makefile              # Common commands
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## Testing
```bash
# Run all tests
make test

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_analysis.py -v
```

## Development

### Code Quality
```bash
# Format code
uv run ruff format src/ main.py

# Lint code
uv run ruff check src/ main.py

# Type check
uv run pyright src/ main.py
```

### Adding New Features

1. Create feature branch
2. Add implementation in `src/`
3. Add tests in `tests/`
4. Update documentation
5. Run `make test` to verify
6. Submit pull request

## Examples

### Example 1: Quick Forecast
```bash
uv run python main.py --symbol AAPL --days 30
```

### Example 2: Interactive Analysis
```python
from src.data import Fetcher, prepare_for_prophet
from src.models import ForecastModel
from src.analysis import ForecastAnalyzer

# Fetch data
fetcher = Fetcher()
data = fetcher.fetch('AAPL', '2023-01-01', '2024-10-31')
prophet_data = prepare_for_prophet(data)

# Train model
model = ForecastModel()
model.train(prophet_data)
forecast = model.predict(periods=90)

# Analyze
analyzer = ForecastAnalyzer()
optimal = analyzer.find_optimal_sell_date(forecast)
print(f"Optimal sell date: {optimal['date']} at ${optimal['price']:.2f}")
```

### Example 3: Compare Multiple Stocks

See `notebooks/01_explore_data.ipynb` for interactive example.

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'prophet'`
```bash
# Solution: Install dependencies
make install
```

**Issue:** `FileNotFoundError: config.yaml`
```bash
# Solution: Ensure config.yaml exists in project root
ls config.yaml
```

**Issue:** Forecast fails for certain stocks
```bash
# Solution: Try different date range or stock symbol
uv run python main.py --symbol MSFT --days 30
```

## Performance

- Typical forecast (90 days): ~10-15 seconds
- Training on 1 year of data: ~5-8 seconds
- Generating plots: ~2-3 seconds

## Roadmap

- [x] Basic forecasting pipeline
- [x] Multiple scenarios
- [x] Volatility analysis
- [x] Beautiful visualizations
- [x] Jupyter notebooks
- [x] Comprehensive tests
- [ ] Multi-stock comparison
- [ ] Portfolio optimization
- [ ] Real-time data streaming
- [ ] Web dashboard

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Facebook Prophet** - Time series forecasting
- **Yahoo Finance** - Stock data API
- **Python Data Science Community** - Amazing ecosystem

## Support

- 📖 Documentation: This README
- 📓 Examples: `notebooks/` directory
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---
