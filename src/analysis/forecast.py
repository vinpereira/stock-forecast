import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def calculate_metrics(data):
    metrics = {
        'mean': data['Close'].mean(),
        'std': data['Close'].std(),
        'min': data['Close'].min(),
        'max': data['Close'].max(),
        'range': data['Close'].max() - data['Close'].min()
    }
    return metrics


class ForecastAnalyzer:
    def __init__(self):
        pass
    
    def get_future_values(self, forecast: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        today = pd.Timestamp.now().normalize()
        future_forecast = forecast[forecast['ds'] > today].copy()
        
        if days > 0 and len(future_forecast) > days:
            future_forecast = future_forecast.head(days)
        
        return future_forecast
    
    def find_optimal_sell_date(self, forecast: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        if start_date is None:
            start_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = forecast['ds'].max().strftime('%Y-%m-%d')
        
        mask = (
            (forecast['ds'] >= pd.Timestamp(start_date)) & 
            (forecast['ds'] <= pd.Timestamp(end_date))
        )
        period_forecast = forecast[mask].copy()
        
        if len(period_forecast) == 0:
            return {
                'date': None,
                'price': None,
                'error': 'No data in specified date range'
            }
        
        max_idx = period_forecast['yhat'].idxmax()
        optimal_row = period_forecast.loc[max_idx]
        
        return {
            'date': optimal_row['ds'].strftime('%Y-%m-%d'),
            'price': float(optimal_row['yhat']),
            'price_optimistic': float(optimal_row['yhat_upper']),
            'price_pessimistic': float(optimal_row['yhat_lower']),
            'confidence_range': float(optimal_row['yhat_upper'] - optimal_row['yhat_lower']),
            'days_from_now': (pd.Timestamp(optimal_row['ds']) - pd.Timestamp.now()).days
        }
    
    def calculate_scenarios(self, forecast: pd.DataFrame, target_date: Optional[str] = None) -> Dict[str, float]:
        if target_date is None:
            target_row = forecast.iloc[-1]
        else:
            target_ts = pd.Timestamp(target_date)
            target_row = forecast[forecast['ds'] == target_ts].iloc[0]
        
        return {
            'date': target_row['ds'].strftime('%Y-%m-%d'),
            'expected': float(target_row['yhat']),
            'optimistic': float(target_row['yhat_upper']),
            'pessimistic': float(target_row['yhat_lower']),
            'range': float(target_row['yhat_upper'] - target_row['yhat_lower']),
            'uncertainty': float(
                (target_row['yhat_upper'] - target_row['yhat_lower']) / target_row['yhat'] * 100
            )
        }
    
    def get_volatility_metrics(self, forecast: pd.DataFrame) -> Dict[str, float]:
        forecast = forecast.copy()
        forecast['ci_width'] = forecast['yhat_upper'] - forecast['yhat_lower']
        forecast['ci_pct'] = (forecast['ci_width'] / forecast['yhat']) * 100
        
        return {
            'avg_uncertainty': float(forecast['ci_pct'].mean()),
            'max_uncertainty': float(forecast['ci_pct'].max()),
            'min_uncertainty': float(forecast['ci_pct'].min()),
            'uncertainty_std': float(forecast['ci_pct'].std()),
            'avg_ci_width': float(forecast['ci_width'].mean()),
            'max_ci_width': float(forecast['ci_width'].max())
        }
    
    def export_to_csv(self, forecast: pd.DataFrame, filename: str, include_components: bool = False) -> str:
        if include_components:
            columns = [
                'ds', 'yhat', 'yhat_lower', 'yhat_upper',
                'trend', 'yearly', 'weekly', 'holidays'
            ]
            columns = [col for col in columns if col in forecast.columns]
            export_df = forecast[columns].copy()
        else:
            export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        export_df.columns = [
            'Date', 
            'Forecast', 
            'Lower_Bound_95%', 
            'Upper_Bound_95%'
        ] if not include_components else export_df.columns
        
        export_df.to_csv(filename, index=False)
        print(f"✓ Forecast exported to: {filename}")
        return filename
    
    def print_summary(self, forecast: pd.DataFrame, current_price: float, symbol: str = "Stock") -> None:
        print("\n" + "="*80)
        print(f"📊 FORECAST SUMMARY - {symbol}")
        print("="*80)
        
        # Current info
        print(f"\n💰 Current Price: ${current_price:.2f}")
        print(f"📅 Current Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        
        # 30-day forecast
        future_30d = self.get_future_values(forecast, days=30)
        if len(future_30d) > 0:
            last_30d = future_30d.iloc[-1]
            change_pct = ((last_30d['yhat'] - current_price) / current_price * 100)
            print(f"\n📈 30-Day Forecast:")
            print(f"   Expected: ${last_30d['yhat']:.2f} ({change_pct:+.2f}%)")
            print(f"   Optimistic: ${last_30d['yhat_upper']:.2f}")
            print(f"   Pessimistic: ${last_30d['yhat_lower']:.2f}")
        
        # Full forecast
        last_forecast = forecast.iloc[-1]
        days_ahead = (last_forecast['ds'] - pd.Timestamp.now()).days
        change_pct = ((last_forecast['yhat'] - current_price) / current_price * 100)
        
        print(f"\n📈 {days_ahead}-Day Forecast:")
        print(f"   Expected: ${last_forecast['yhat']:.2f} ({change_pct:+.2f}%)")
        print(f"   Optimistic: ${last_forecast['yhat_upper']:.2f}")
        print(f"   Pessimistic: ${last_forecast['yhat_lower']:.2f}")
        
        uncertainty = ((last_forecast['yhat_upper'] - last_forecast['yhat_lower']) / last_forecast['yhat'] * 100)
        print(f"   Uncertainty: ±{uncertainty:.1f}%")
        
        # Best sell date
        optimal = self.find_optimal_sell_date(forecast)
        if optimal['date']:
            print(f"\n🎯 Optimal Sell Date: {optimal['date']}")
            print(f"   Expected Price: ${optimal['price']:.2f}")
            print(f"   Days from now: {optimal['days_from_now']}")
        
        # Volatility
        volatility = self.get_volatility_metrics(forecast)
        print(f"\n📊 Average Uncertainty: ±{volatility['avg_uncertainty']:.1f}%")
        
        print("\n" + "="*80 + "\n")
