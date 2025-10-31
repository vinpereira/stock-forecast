# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, Any


class ForecastPlotter:
    def __init__(self, output_dir: str = './outputs', figsize: tuple[int, int] = (16, 8), dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_forecast(self, model: Any, forecast: pd.DataFrame, symbol: str, show_annotations: bool = True) -> str:
        print("📊 Creating forecast plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data (black dots)
        if hasattr(model, 'history') and model.history is not None:
            history = model.history
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
        
        # Plot UPPER LIMIT (green dashed line)
        ax.plot(
            forecast['ds'], 
            forecast['yhat_upper'], 
            'g--', 
            linewidth=2, 
            label='Optimistic Scenario (95%)', 
            alpha=0.8
        )
        
        # Plot LOWER LIMIT (red dashed line)
        ax.plot(
            forecast['ds'], 
            forecast['yhat_lower'], 
            'r--', 
            linewidth=2, 
            label='Pessimistic Scenario (95%)', 
            alpha=0.8
        )
        
        # Fill confidence interval
        # Convert to numpy arrays for matplotlib compatibility
        x_data = np.array(forecast['ds'].values, dtype='datetime64[ns]')
        y1_data = np.array(forecast['yhat_lower'].values, dtype=float)
        y2_data = np.array(forecast['yhat_upper'].values, dtype=float)
        
        ax.fill_between(  # type: ignore
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
            self._add_value_annotations(ax, forecast, model)
        
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
        
        # Save plot
        plot_path = self.output_dir / f'forecast_{symbol}.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Forecast plot saved: {plot_path}")
        plt.close()
        
        return str(plot_path)
    
    def plot_components(self, model: Any, forecast: pd.DataFrame, symbol: str) -> str:
        print("📊 Creating components plot...")
        
        # Access the underlying Prophet model
        if hasattr(model, 'model') and model.model is not None:
            prophet_model = model.model
        else:
            raise ValueError("Model has not been trained yet")
        
        # Use Prophet's built-in plot_components
        fig = prophet_model.plot_components(forecast, figsize=(14, 10))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'forecast_components_{symbol}.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Components plot saved: {plot_path}")
        plt.close()
        
        return str(plot_path)
    
    def _add_value_annotations(self, ax: Axes, forecast: pd.DataFrame, model: Any) -> None:
        # Get last forecast (1 year in future)
        last_idx = len(forecast) - 1
        last_date = forecast['ds'].iloc[last_idx]
        
        expected_value = forecast['yhat'].iloc[last_idx]
        optimistic_value = forecast['yhat_upper'].iloc[last_idx]
        pessimistic_value = forecast['yhat_lower'].iloc[last_idx]
        
        # Annotation for OPTIMISTIC (green, at top)
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
        
        # Add summary text box - À DIREITA da legenda
        if hasattr(model, 'history') and model.history is not None:
            current_value = model.history['y'].iloc[-1]
            variation = ((expected_value - current_value) / current_value) * 100
            
            summary_text = (
                f'1 YEAR FORECAST (365 days)\n'
                f'━━━━━━━━━━━━━━━━━━━━━━\n'
                f'Current Value: ${current_value:.2f}\n'
                f'Expected: ${expected_value:.2f} ({variation:+.1f}%)\n'
                f'Optimistic: ${optimistic_value:.2f}\n'
                f'Pessimistic: ${pessimistic_value:.2f}\n'
                f'Range: ${optimistic_value - pessimistic_value:.2f}'
            )
            
            # Posição: canto superior DIREITO (ao lado da legenda)
            ax.text(
                0.27, 0.98,
                summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                family='monospace'
            )
    
    def create_comparison_plot(self, forecasts: Dict[str, pd.DataFrame], symbols: list[str], title: str = "Stock Comparison") -> str:
        print(f"📊 Creating comparison plot for {len(symbols)} stocks...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        for i, symbol in enumerate(symbols):
            if symbol in forecasts:
                forecast = forecasts[symbol]
                color = colors[i % len(colors)]
                
                ax.plot(
                    forecast['ds'],
                    forecast['yhat'],
                    label=symbol,
                    color=color,
                    linewidth=2
                )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Price', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'comparison.png'
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {plot_path}")
        plt.close()
        
        return str(plot_path)


def plot_forecast_simple(model: Any, forecast: pd.DataFrame, symbol: str, output_dir: str = './outputs') -> tuple[str, str]:
    plotter = ForecastPlotter(output_dir=output_dir)
    
    forecast_path = plotter.plot_forecast(model, forecast, symbol)
    components_path = plotter.plot_components(model, forecast, symbol)
    
    return forecast_path, components_path
