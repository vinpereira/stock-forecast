import os
import sys
import argparse
from datetime import datetime
from src.data import Fetcher, prepare_for_prophet
from src.models import ForecastModel
from src.analysis import ForecastAnalyzer
from src.visualization import plot_forecast
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Stock price forecasting')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--days', type=int, help='Forecast days')
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 STOCK PRICE FORECASTING WITH PROPHET")
    print("=" * 80)
    
    try:
        # Load config
        print("\n🔧 Step 1: Loading Configuration...")
        cfg = load_config()
        
        symbol = args.symbol if args.symbol else cfg['stock']['symbol']
        start = cfg['stock']['start']
        end = cfg['stock']['end']
        forecast_days = args.days if args.days else cfg['forecast']['days']
        
        # Handle 'today' keyword
        if end.lower() == 'today':
            end = datetime.now().strftime('%Y-%m-%d')
            print(f"   ℹ️  Using today's date: {end}")
        
        print(f"   Symbol: {symbol}")
        print(f"   Period: {start} to {end}")
        print(f"   Forecast: {forecast_days} days")
        
        # Fetch
        print(f"\n📥 Step 2: Fetching Data for {symbol}...")
        f = Fetcher()
        data = f.fetch(symbol, start, end)
        print(f"   ✓ Fetched {len(data)} records")
        
        # Preprocess
        print("\n🧹 Step 3: Preprocessing Data...")
        prophet_data = prepare_for_prophet(data)
        
        stats = {
            'min': prophet_data['y'].min(),
            'max': prophet_data['y'].max(),
            'mean': prophet_data['y'].mean()
        }
        print(f"   Price range: ${stats['min']:.2f} - ${stats['max']:.2f}")
        print(f"   Mean price: ${stats['mean']:.2f}")
        
        # Train
        print("\n🤖 Step 4: Training Prophet Model...")
        model = ForecastModel()
        model.train(prophet_data)
        print(f"   ✓ Trained on {len(prophet_data)} samples")
        
        # Forecast
        print(f"\n🔮 Step 5: Generating Forecasts...")
        forecast = model.predict(periods=forecast_days)
        
        # Visualize
        print("\n📊 Step 6: Creating Visualizations...")
        output_dir = cfg.get('output', {}).get('directory', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plot_forecast(forecast, prophet_data)
        png_file = f'{output_dir}/forecast_{symbol}.png'
        fig.savefig(png_file)
        print(f"   ✓ Saved plot: {png_file}")
        
        # Analyze
        print("\n📈 Step 7: Analyzing Results...")
        analyzer = ForecastAnalyzer()
        
        current_price = float(prophet_data['y'].iloc[-1])
        analyzer.print_summary(forecast, current_price, symbol)
        
        # Export CSV
        csv_file = f'{output_dir}/forecast_{symbol}.csv'
        analyzer.export_to_csv(forecast, csv_file, include_components=True)
        
        # Success
        print("=" * 80)
        print("✅ FORECASTING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📁 Output files saved to: {output_dir}/")
        print(f"   • Forecast plot: forecast_{symbol}.png")
        print(f"   • CSV data: forecast_{symbol}.csv")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
