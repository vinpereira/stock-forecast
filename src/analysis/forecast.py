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
