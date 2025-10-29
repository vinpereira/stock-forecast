import os
from src.data import Fetcher, prepare_for_prophet
from src.models import ForecastModel
from src.analysis import calculate_metrics
from src.visualization import plot_forecast
from src.utils.config import load_config

def main():
    cfg = load_config()
    
    # Fetch
    symbol = cfg['stock']['symbol']
    start = cfg['stock']['start']
    end = cfg['stock']['end']
    
    print(f"Fetching {symbol}...")
    f = Fetcher()
    data = f.fetch(symbol, start, end)
    print(f"Got {len(data)} rows")
    
    # Analyze
    print("\nData metrics:")
    metrics = calculate_metrics(data)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Prepare
    print("\nPreparing data for Prophet...")
    prophet_data = prepare_for_prophet(data)
    
    # Train
    print("Training model...")
    model = ForecastModel()
    model.train(prophet_data)
    
    # Forecast
    forecast_days = cfg['forecast']['days']
    print(f"\nGenerating {forecast_days}-day forecast...")
    forecast = model.predict(periods=forecast_days)
    
    print("\nLast 10 predictions:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
    
    # Plot
    print("\nGenerating plot...")
    fig = plot_forecast(forecast, prophet_data)
    
    # Create outputs dir if needed
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/forecast.png')
    print("Saved to outputs/forecast.png")

if __name__ == '__main__':
    main()