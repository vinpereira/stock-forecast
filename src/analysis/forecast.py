# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportReturnType=false

import pandas as pd
from typing import Optional, Dict, Any

def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        'mean': float(data['Close'].mean()),
        'std': float(data['Close'].std()),
        'min': float(data['Close'].min()),
        'max': float(data['Close'].max()),
        'range': float(data['Close'].max() - data['Close'].min())
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
        # Handle empty forecast
        if len(forecast) == 0:
            return {
                'date': None,
                'price': None,
                'error': 'No forecast data available'
            }
        
        if start_date is None:
            start_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        if end_date is None:
            max_date = forecast['ds'].max()
            # Handle NaN case
            if pd.isna(max_date):
                return {
                    'date': None,
                    'price': None,
                    'error': 'No valid dates in forecast'
                }
            end_date = max_date.strftime('%Y-%m-%d')
        
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
    
    def generate_scenarios(self, forecast: pd.DataFrame, target_date: Optional[str] = None) -> Dict[str, Any]:
        if target_date is None:
            target_date = forecast['ds'].max().strftime('%Y-%m-%d')
        
        target_data = forecast[forecast['ds'] == pd.Timestamp(target_date)]
        
        if len(target_data) == 0:
            return {
                'error': f'No forecast data for {target_date}'
            }
        
        row = target_data.iloc[0]
        
        scenarios: Dict[str, Any] = {
            'optimistic': {
                'price': float(row['yhat_upper']),
                'probability': 0.025,
                'description': 'Best case scenario (97.5th percentile)'
            },
            'expected': {
                'price': float(row['yhat']),
                'probability': 0.50,
                'description': 'Most likely outcome'
            },
            'pessimistic': {
                'price': float(row['yhat_lower']),
                'probability': 0.025,
                'description': 'Worst case scenario (2.5th percentile)'
            }
        }
        
        return scenarios
    
    def calculate_volatility(self, forecast: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        future = self.get_future_values(forecast, days=window)
        
        if len(future) == 0:
            return {'error': 'No future data available'}
        
        # Convert to float explicitly
        yhat_values = future['yhat'].astype(float)
        yhat_upper_values = future['yhat_upper'].astype(float)
        yhat_lower_values = future['yhat_lower'].astype(float)
        
        volatility: Dict[str, Any] = {
            'std_dev': float(yhat_values.std()),
            'coefficient_of_variation': float(yhat_values.std() / yhat_values.mean()),
            'avg_confidence_range': float((yhat_upper_values - yhat_lower_values).mean()),
            'max_confidence_range': float((yhat_upper_values - yhat_lower_values).max()),
            'min_confidence_range': float((yhat_upper_values - yhat_lower_values).min())
        }
        
        return volatility
    
    def print_summary(self, forecast: pd.DataFrame, current_price: float, symbol: str) -> None:
        print(f"\n{'='*80}")
        print(f"📊 FORECAST SUMMARY - {symbol}")
        print(f"{'='*80}")
        
        # Current info
        print(f"\n💰 Current Price: ${current_price:.2f}")
        
        # Future predictions
        future_30 = self.get_future_values(forecast, days=30)
        future_90 = self.get_future_values(forecast, days=90)
        
        if len(future_30) > 0:
            last_30 = future_30.iloc[-1]
            print(f"\n📈 30-Day Forecast:")
            print(f"   Expected:    ${float(last_30['yhat']):.2f}")
            print(f"   Optimistic:  ${float(last_30['yhat_upper']):.2f}")
            print(f"   Pessimistic: ${float(last_30['yhat_lower']):.2f}")
            
            change_30 = ((float(last_30['yhat']) - current_price) / current_price) * 100
            print(f"   Change: {change_30:+.2f}%")
        
        if len(future_90) > 0:
            last_90 = future_90.iloc[-1]
            print(f"\n📈 90-Day Forecast:")
            print(f"   Expected:    ${float(last_90['yhat']):.2f}")
            print(f"   Optimistic:  ${float(last_90['yhat_upper']):.2f}")
            print(f"   Pessimistic: ${float(last_90['yhat_lower']):.2f}")
            
            change_90 = ((float(last_90['yhat']) - current_price) / current_price) * 100
            print(f"   Change: {change_90:+.2f}%")
        
        # Optimal sell date
        optimal = self.find_optimal_sell_date(forecast)
        if 'error' not in optimal:
            print(f"\n🎯 Optimal Sell Date: {optimal['date']}")
            print(f"   Expected Price: ${optimal['price']:.2f}")
            print(f"   Days from now: {optimal['days_from_now']}")
        
        # Volatility
        volatility = self.calculate_volatility(forecast, window=90)
        if 'error' not in volatility:
            print(f"\n📊 Volatility (90-day):")
            print(f"   Std Dev: ${volatility['std_dev']:.2f}")
            print(f"   Avg Confidence Range: ${volatility['avg_confidence_range']:.2f}")
        
        print(f"\n{'='*80}\n")
    
    def export_to_csv(self, forecast: pd.DataFrame, output_path: str, include_components: bool = False) -> None:
        if include_components:
            # Include all columns
            forecast.to_csv(output_path, index=False)
        else:
            # Only essential columns
            essential = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            forecast[essential].to_csv(output_path, index=False)
        
        print(f"✅ Forecast exported to: {output_path}")
