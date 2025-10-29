import os
import argparse
from src.data import Fetcher, prepare_for_prophet
from src.models import ForecastModel
from src.analysis import calculate_metrics
from src.visualization import plot_forecast
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Stock price forecasting')
    parser.add_argument('--symbol', type=str, help='Stock symbol (overrides config)')
    parser.add_argument('--days', type=int, help='Forecast days (overrides config)')
    args = parser.parse_args()
    
    cfg = load_config()
    
    # Use CLI args if provided, otherwise use config
    symbol = args.symbol if args.symbol else cfg['stock']['symbol']
    start = cfg['stock']['start']
    end = cfg['stock']['end']
    forecast_days = args.days if args.days else cfg['forecast']['days']
    
    # Fetch
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
    print(f"\nGenerating {forecast_days}-day forecast...")
    forecast = model.predict(periods=forecast_days)
    
    print("\nLast 10 predictions:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
    
    # Create outputs dir if needed
    os.makedirs('outputs', exist_ok=True)
    
    # Save forecast to CSV
    csv_file = f'outputs/forecast_{symbol}.csv'
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(csv_file, index=False)
    print(f"\nSaved forecast to {csv_file}")
    
    # Plot
    print("Generating plot...")
    fig = plot_forecast(forecast, prophet_data)
    png_file = f'outputs/forecast_{symbol}.png'
    fig.savefig(png_file)
    print(f"Saved plot to {png_file}")

if __name__ == '__main__':
    main()
