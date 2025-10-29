import matplotlib.pyplot as plt

def plot_forecast(forecast, actual_data=None):
    """Plot forecast with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        alpha=0.3,
        color='blue'
    )
    
    # Plot actual if provided
    if actual_data is not None:
        ax.plot(actual_data['ds'], actual_data['y'], 
                label='Actual', color='black', marker='.')
    
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Price Forecast')
    ax.grid(True, alpha=0.3)
    
    return fig
