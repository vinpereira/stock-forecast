import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Any

def plot_forecast(model: Any, forecast: pd.DataFrame, symbol: str, output_dir: str = './outputs', dpi: int = 300, show_annotations: bool = True) -> str:
    print("📊 Creating forecast plot...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical data (black dots)
    if hasattr(model, 'model') and model.model is not None:
        if hasattr(model.model, 'history') and model.model.history is not None:
            history = model.model.history
            ax.plot(
                history['ds'], 
                history['y'], 
                'k.', 
                label='Historical Data', 
                markersize=3, 
                alpha=0.5
            )
    
    # Plot FORECAST (blue line)
    ax.plot(
        forecast['ds'], 
        forecast['yhat'], 
        'b-', 
        linewidth=2.5, 
        label='Forecast (Expected)', 
        alpha=0.9
    )
    
    # Plot UPPER LIMIT (green dashed)
    ax.plot(
        forecast['ds'], 
        forecast['yhat_upper'], 
        'g--', 
        linewidth=2, 
        label='Optimistic Scenario (95%)', 
        alpha=0.8
    )
    
    # Plot LOWER LIMIT (red dashed)
    ax.plot(
        forecast['ds'], 
        forecast['yhat_lower'], 
        'r--', 
        linewidth=2, 
        label='Pessimistic Scenario (95%)', 
        alpha=0.8
    )
    
    # Fill confidence interval
    x_data = np.array(forecast['ds'].values, dtype='datetime64[ns]')
    y1_data = np.array(forecast['yhat_lower'].values, dtype=float)
    y2_data = np.array(forecast['yhat_upper'].values, dtype=float)
    
    ax.fill_between(
        x_data, 
        y1_data, 
        y2_data,
        alpha=0.15, 
        color='gray', 
        label='95% Confidence Interval'
    )
    
    # Add vertical line for "today"
    today = pd.Timestamp.now()
    today_num = mdates.date2num(today.to_pydatetime())
    ax.axvline(
        x=today_num, 
        color='orange', 
        linestyle=':', 
        linewidth=2.5, 
        label=f'Today ({today.strftime("%Y-%m-%d")})', 
        alpha=0.8
    )
    
    # Add annotations if requested
    if show_annotations:
        _add_value_annotations(ax, forecast, model)
    
    # Styling
    ax.set_title(
        f'Price Forecast - {symbol}\nProphet Model with Confidence Intervals', 
        fontsize=16, 
        fontweight='bold', 
        pad=20
    )
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / f'forecast_{symbol}.png'
    
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Forecast plot saved: {plot_path}")
    plt.close()
    
    return str(plot_path)


def _add_value_annotations(ax: plt.Axes, forecast: pd.DataFrame, model: Any) -> None:
    last_idx = len(forecast) - 1
    last_date = forecast['ds'].iloc[last_idx]
    
    expected_value = forecast['yhat'].iloc[last_idx]
    optimistic_value = forecast['yhat_upper'].iloc[last_idx]
    pessimistic_value = forecast['yhat_lower'].iloc[last_idx]
    
    # Annotation for OPTIMISTIC (green, top)
    ax.annotate(
        f'Optimistic: ${optimistic_value:.2f}',
        xy=(last_date, optimistic_value),
        xytext=(10, 10), 
        textcoords='offset points',
        fontsize=11, 
        fontweight='bold', 
        color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='green', lw=2)
    )
    
    # Annotation for EXPECTED (blue, middle)
    ax.annotate(
        f'Expected: ${expected_value:.2f}',
        xy=(last_date, expected_value),
        xytext=(10, -5), 
        textcoords='offset points',
        fontsize=11, 
        fontweight='bold', 
        color='darkblue',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='blue', lw=2)
    )
    
    # Annotation for PESSIMISTIC (red, bottom)
    ax.annotate(
        f'Pessimistic: ${pessimistic_value:.2f}',
        xy=(last_date, pessimistic_value),
        xytext=(10, -20), 
        textcoords='offset points',
        fontsize=11, 
        fontweight='bold', 
        color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='red', lw=2)
    )
    
    # Add summary text box
    if hasattr(model, 'model') and model.model is not None:
        if hasattr(model.model, 'history') and model.model.history is not None:
            current_value = model.model.history['y'].iloc[-1]
            variation = ((expected_value - current_value) / current_value) * 100
            
            days_ahead = (last_date - pd.Timestamp.now()).days
            
            summary_text = (
                f'{days_ahead}-DAY FORECAST\n'
                f'━━━━━━━━━━━━━━━━━━━━\n'
                f'Current: ${current_value:.2f}\n'
                f'Expected: ${expected_value:.2f} ({variation:+.1f}%)\n'
                f'Optimistic: ${optimistic_value:.2f}\n'
                f'Pessimistic: ${pessimistic_value:.2f}\n'
                f'Range: ${optimistic_value - pessimistic_value:.2f}'
            )
            
            ax.text(
                0.02, 0.98, 
                summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                family='monospace'
            )


def plot_components(model: Any, forecast: pd.DataFrame, symbol: str, output_dir: str = './outputs', dpi: int = 300) -> str:
    print("📊 Creating components plot...")
    
    # Access underlying Prophet model
    if hasattr(model, 'model') and model.model is not None:
        prophet_model = model.model
    else:
        raise ValueError("Model has not been trained yet")
    
    # Use Prophet's built-in plot_components
    fig = prophet_model.plot_components(forecast, figsize=(14, 10))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / f'forecast_components_{symbol}.png'
    
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Components plot saved: {plot_path}")
    plt.close()
    
    return str(plot_path)
