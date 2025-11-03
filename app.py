import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

from src.data import Fetcher, prepare_for_prophet
from src.models import ForecastModel
from src.analysis import ForecastAnalyzer
from src.visualization.plotter import ForecastPlotter

# Page config
st.set_page_config(
    page_title="Stock Forecast",
    page_icon="📊",
    layout="wide"
)

# Cache functions for performance
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, start_date: str, end_date: str):
    fetcher = Fetcher()
    
    if not fetcher.validate_symbol(symbol):
        return None
    
    data = fetcher.fetch(symbol, start_date, end_date)
    return data


@st.cache_data(ttl=3600)
def train_and_forecast(data, symbol: str, days: int):
    # Prepare data
    prophet_data = prepare_for_prophet(data)
    
    # Train model
    model = ForecastModel()
    model.train(prophet_data)
    
    # Generate forecast
    forecast = model.predict(periods=days)
    
    return prophet_data, model, forecast


def create_forecast_plot(prophet_data, forecast, symbol):
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical data (black dots)
    ax.plot(
        prophet_data['ds'], 
        prophet_data['y'], 
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
    today_num = float(mdates.date2num(today.to_pydatetime()))
    ax.axvline(
        x=today_num, 
        color='orange', 
        linestyle=':', 
        linewidth=2.5, 
        label=f'Today ({today.strftime("%Y-%m-%d")})', 
        alpha=0.8
    )
    
    # Add annotations for final values
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
    
    return fig


def analyze_range_quality(range_value: float, expected_price: float):
    range_pct = (range_value / expected_price) * 100
    
    if range_pct < 15:
        return "🟢 Excelente", "Alta confiança na previsão. Range estreito indica previsibilidade.", "success"
    elif range_pct < 30:
        return "🟡 Moderado", "Confiança média. Range moderado é normal para médio prazo.", "warning"
    else:
        return "🔴 Alto", "Baixa confiança. Range amplo indica alta incerteza.", "error"


# Title
st.title("📊 Stock Price Forecasting with Prophet")
st.markdown("*Professional stock price forecasting powered by Facebook Prophet*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock symbol
    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Forecast period
    forecast_days = st.slider(
        "Forecast Days",
        min_value=7,
        max_value=365,
        value=90,
        step=7,
        help="Number of days to forecast into the future"
    )
    
    # Generate button
    forecast_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
    
    st.divider()
    
    # Info
    with st.expander("ℹ️ About"):
        st.markdown("""
        This app uses **Facebook Prophet** to forecast stock prices.
        
        **Features:**
        - Historical data visualization
        - Multi-scenario forecasting
        - Optimal sell date recommendation
        - Volatility analysis
        
        **Note:** This is for educational purposes only.
        Not financial advice!
        """)

# Main content
if forecast_btn:
    if not symbol:
        st.error("⚠️ Please enter a stock symbol!")
        st.stop()
    
    # Fetch data
    with st.spinner(f"📥 Fetching data for {symbol}..."):
        data = fetch_stock_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    if data is None or len(data) == 0:
        st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
        st.stop()
    
    st.success(f"✅ Fetched {len(data)} records for {symbol}")
    
    # Train and forecast
    with st.spinner("🤖 Training Prophet model and generating forecast..."):
        prophet_data, model, forecast = train_and_forecast(data, symbol, forecast_days)
    
    st.success(f"✅ Forecast generated for {forecast_days} days!")
    
    # Analyze
    analyzer = ForecastAnalyzer()
    current_price = float(prophet_data['y'].iloc[-1])
    
    # Metrics
    st.subheader("📈 Key Metrics")
    
    future_30 = analyzer.get_future_values(forecast, days=min(30, forecast_days))
    optimal = analyzer.find_optimal_sell_date(forecast)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            help="Latest closing price"
        )
    
    with col2:
        if len(future_30) > 0:
            expected_30 = float(future_30.iloc[-1]['yhat'])
            change_30 = ((expected_30 - current_price) / current_price) * 100
            st.metric(
                "Expected (30d)",
                f"${expected_30:.2f}",
                f"{change_30:+.2f}%",
                help="Expected price in 30 days"
            )
    
    with col3:
        if 'error' not in optimal:
            st.metric(
                "Optimal Price",
                f"${optimal['price']:.2f}",
                help="Best predicted price in forecast period"
            )
    
    with col4:
        if 'error' not in optimal:
            st.metric(
                "Optimal Date",
                optimal['date'],
                f"{optimal['days_from_now']} days",
                help="Best date to sell"
            )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "🔍 Components", "📋 Data", "📉 Analysis"])
    
    with tab1:
        st.subheader("Price Forecast")
        
        # Create styled plot (SEM forecast_days)
        fig = create_forecast_plot(prophet_data, forecast, symbol)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("Forecast Components")
        
        # Use Prophet's plot_components
        if hasattr(model, 'model') and model.model is not None:
            fig_components = model.model.plot_components(forecast)
            st.pyplot(fig_components)
            plt.close()
        else:
            st.error("Model not available for component plotting")
    
    with tab3:
        st.subheader("Forecast Data")
        
        # Show future data
        future_df = analyzer.get_future_values(forecast, days=forecast_days)
        display_df = future_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        display_df.columns = ['Date', 'Expected', 'Lower Bound', 'Upper Bound']
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df.style.format({
                'Expected': '${:.2f}',
                'Lower Bound': '${:.2f}',
                'Upper Bound': '${:.2f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"forecast_{symbol}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Scenarios")
            
            # Get scenarios for 30 days AND full forecast period
            periods_to_show = []
            
            # Always show 30-day if available
            if len(future_30) > 0:
                periods_to_show.append(('30-day', future_30.iloc[-1]))
            
            # Show full forecast period if different from 30 days
            if forecast_days > 30:
                future_full = analyzer.get_future_values(forecast, days=forecast_days)
                if len(future_full) > 0:
                    periods_to_show.append((f'{forecast_days}-day', future_full.iloc[-1]))
            
            # Display scenarios for each period
            for period_label, last_row in periods_to_show:
                target_date = last_row['ds'].strftime('%Y-%m-%d')
                scenarios = analyzer.generate_scenarios(forecast, target_date)
                
                optimistic = scenarios['optimistic']['price']
                expected = scenarios['expected']['price']
                pessimistic = scenarios['pessimistic']['price']
                range_value = optimistic - pessimistic
                
                st.markdown(f"**{period_label} outlook ({target_date}):**")
                st.markdown(f"- **Optimistic:** ${optimistic:.2f}")
                st.markdown(f"- **Expected:** ${expected:.2f}")
                st.markdown(f"- **Pessimistic:** ${pessimistic:.2f}")
                st.markdown(f"- **Range:** ${range_value:.2f}")
                
                # Range analysis
                quality, explanation, alert_type = analyze_range_quality(range_value, expected)
                
                if alert_type == "success":
                    st.success(f"**{quality}** - {explanation}")
                elif alert_type == "warning":
                    st.warning(f"**{quality}** - {explanation}")
                else:
                    st.error(f"**{quality}** - {explanation}")
                
                # Add separator between periods
                if period_label == '30-day' and len(periods_to_show) > 1:
                    st.markdown("---")
        
        with col2:
            st.markdown("### 📊 Volatility")
            
            volatility = analyzer.calculate_volatility(forecast, window=min(90, forecast_days))
            
            if 'error' not in volatility:
                st.markdown(f"**Standard Deviation:** ${volatility['std_dev']:.2f}")
                st.markdown(f"**Coefficient of Variation:** {volatility['coefficient_of_variation']:.4f}")
                st.markdown(f"**Avg Confidence Range:** ${volatility['avg_confidence_range']:.2f}")
                
                # Range interpretation
                st.markdown("---")
                st.markdown("**📖 Interpretação:**")
                st.markdown("""
                - **Range < 15% do preço:** Alta confiança 🟢
                - **Range 15-30%:** Confiança moderada 🟡
                - **Range > 30%:** Baixa confiança 🔴
                
                *Ranges maiores indicam maior incerteza na previsão.*
                """)

else:
    # Welcome screen
    st.info("👈 Configure your forecast settings in the sidebar and click **Generate Forecast** to begin!")
    
    # Show example
    st.subheader("📖 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Configure
        - Enter stock symbol (e.g., AAPL)
        - Select date range
        - Choose forecast period
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Generate
        - Click "Generate Forecast"
        - Wait for model training
        - View results
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Analyze
        - Review forecast chart
        - Check components
        - Download data
        """)
    
    # Popular stocks
    st.subheader("💡 Popular Stocks to Try")
    
    stocks = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corp."
    }
    
    cols = st.columns(3)
    for i, (ticker, name) in enumerate(stocks.items()):
        with cols[i % 3]:
            st.markdown(f"**{ticker}** - {name}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Made with ❤️ using Streamlit and Prophet | 
    <a href='https://github.com/yourusername/stock-forecast'>GitHub</a> | 
    ⚠️ Not financial advice
</div>
""", unsafe_allow_html=True)
