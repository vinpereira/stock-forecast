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

## Example
```bash
# Fetch AAPL data and forecast 90 days
uv run python main.py
```
