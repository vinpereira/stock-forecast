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

def plot_components(model, forecast, symbol, output_dir='./outputs'):
    print("📊 Creating components plot...")
    
    if hasattr(model, 'model') and model.model is not None:
        prophet_model = model.model
    else:
        raise ValueError("Model not trained")
    
    fig = prophet_model.plot_components(forecast, figsize=(14, 10))
    
    plt.tight_layout()
    
    plot_path = f'{output_dir}/forecast_components_{symbol}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Components plot saved: {plot_path}")
    plt.close()
    
    return plot_path
