# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false

from prophet import Prophet
import pandas as pd
import logging
from typing import Optional, Dict, Any

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class ForecastModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model: Optional[Prophet] = None
        self.trained = False
    
    def train(self, data: pd.DataFrame) -> None:
        print("🤖 Training Prophet model...")
        
        # Create model with config
        self.model = Prophet(
            changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10),
            holidays_prior_scale=self.config.get('holidays_prior_scale', 15),
            weekly_seasonality=self.config.get('weekly_seasonality', True),
            yearly_seasonality=self.config.get('yearly_seasonality', True),
            daily_seasonality=self.config.get('daily_seasonality', False),
        )
        
        # Add holidays if configured
        country = self.config.get('country_holidays')
        if country and self.model:
            try:
                self.model.add_country_holidays(country_name=country)
                print(f"   Added {country} holidays")
            except Exception as e:
                print(f"   ⚠️  Could not add holidays: {e}")
        
        if self.model:
            self.model.fit(data)
            self.trained = True
            print("✅ Model trained successfully!")
    
    def predict(self, periods: int = 365, freq: str = 'D') -> pd.DataFrame:
        if not self.trained or not self.model:
            raise RuntimeError("Model must be trained first")
        
        print(f"🔮 Generating {periods}-day forecast...")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        print(f"✅ Forecast generated!")
        return forecast
    
    def get_components(self, forecast: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if not self.trained or not self.model:
            raise RuntimeError("Model must be trained first")
        
        components = {}
        
        if 'trend' in forecast.columns:
            components['trend'] = forecast[['ds', 'trend']].copy()
        if 'yearly' in forecast.columns:
            components['yearly'] = forecast[['ds', 'yearly']].copy()
        if 'weekly' in forecast.columns:
            components['weekly'] = forecast[['ds', 'weekly']].copy()
        if 'holidays' in forecast.columns:
            components['holidays'] = forecast[['ds', 'holidays']].copy()
        
        return components
    
    def get_changepoints(self) -> Optional[pd.DataFrame]:
        if not self.trained or not self.model:
            return None
        
        if hasattr(self.model, 'changepoints'):
            changepoints = self.model.changepoints
            if len(changepoints) > 0:
                return pd.DataFrame({'changepoint': changepoints})
        
        return None
