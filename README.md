# Stock Forecast

Stock price forecasting with Prophet.

## Setup
```bash
uv sync
```

## Configuration

Edit `config.yaml`:
- Change `symbol` to any stock ticker
- Adjust date range
- Set forecast period (days)

## Usage

### Basic
```bash
uv run python main.py
```

### With CLI arguments
```bash
# Override symbol
uv run python main.py --symbol GOOGL

# Override forecast days
uv run python main.py --days 30

# Both
uv run python main.py --symbol MSFT --days 60
```

## Tests
```bash
uv run pytest
uv run pytest -v
```

## Output

- Console: metrics and predictions
- CSV: `outputs/forecast_<SYMBOL>.csv`
- Plot: `outputs/forecast_<SYMBOL>.png`

## Examples
```bash
# Forecast AAPL for 90 days
uv run python main.py

# Forecast GOOGL for 30 days
uv run python main.py --symbol GOOGL --days 30

# Run tests
uv run pytest -v
```
