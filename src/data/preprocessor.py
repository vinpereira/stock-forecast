import pandas as pd
import logging

logger = logging.getLogger(__name__)

def prepare_for_prophet(df):
    """Convert stock data to Prophet format."""
    print("🧹 Preparing data for Prophet...")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df_reset = df.reset_index()
    
    if 'Date' not in df_reset.columns and 'date' not in df_reset.columns:
        if df_reset.columns[0].lower() in ['index', 'timestamp']:
            df_reset.rename(columns={df_reset.columns[0]: 'Date'}, inplace=True)
    
    date_col = 'Date' if 'Date' in df_reset.columns else 'date'
    
    result = pd.DataFrame({
        'ds': pd.to_datetime(df_reset[date_col]),
        'y': df_reset['close'].values
    })
    
    result = result.dropna()
    
    print(f"✅ Prepared {len(result)} rows for Prophet")
    print(f"   Date range: {result['ds'].min()} to {result['ds'].max()}")
    
    return result
