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
