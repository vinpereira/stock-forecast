from prophet import Prophet
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class ForecastModel:
    """Prophet forecasting model."""
    
    def __init__(self):
        self.model = Prophet()
    
    def train(self, data):
        print("🤖 Training Prophet model...")
        self.model.fit(data)
        print("✅ Model trained successfully!")
    
    def predict(self, periods):
        print(f"🔮 Generating {periods}-day forecast...")
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        print(f"✅ Forecast generated!")
        return forecast
