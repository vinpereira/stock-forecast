from prophet import Prophet

class ForecastModel:
    def __init__(self):
        self.model = Prophet()
    
    def train(self, data):
        self.model.fit(data)
    
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast
