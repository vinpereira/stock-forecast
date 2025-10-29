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

## Run
```bash
uv run python main.py
```

## Tests
```bash
uv run pytest
```

## Output

- Console: metrics and predictions
- File: `outputs/forecast.png`

## Example
```bash
# Forecast AAPL for 90 days
uv run python main.py

# Run tests
uv run pytest -v
```
