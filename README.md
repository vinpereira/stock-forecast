# Stock Forecast

Stock price forecasting with Prophet.

## Setup
```bash
uv sync
```

## Usage

### Basic
```bash
uv run python main.py
```

### With arguments
```bash
uv run python main.py --symbol GOOGL --days 30
```

## Output

The system generates:
- 📊 Forecast summary with scenarios
- 📈 Forecast plot (`outputs/forecast_SYMBOL.png`)
- 📁 CSV data (`outputs/forecast_SYMBOL.csv`)
- 🎯 Optimal sell date recommendation

## Features

- ✅ Prophet time series forecasting
- ✅ Multiple scenarios (optimistic/expected/pessimistic)
- ✅ Optimal sell date finder
- ✅ Volatility analysis
- ✅ Beautiful visualizations
- ✅ CSV export with components

## Tests
```bash
uv run pytest -v
```
