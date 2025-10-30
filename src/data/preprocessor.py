import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    possible_names = ['date', 'ds', 'datetime', 'timestamp', 'time']
    
    for col in df.columns:
        if str(col).lower() in possible_names:
            return str(col)
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    
    return None

def _find_price_column(df: pd.DataFrame, price_column: str = 'close') -> Optional[str]:
    if price_column in df.columns:
        return price_column
    
    for col in df.columns:
        if str(col).lower() == price_column.lower():
            return str(col)
    
    return None

def prepare_for_prophet(df):
    print("🧹 Preparing data for Prophet...")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df_reset = df.reset_index()
    
    # Find date column
    date_col = _find_date_column(df_reset)
    if date_col is None:
        if df_reset.columns[0].lower() in ['index', 'timestamp']:
            df_reset.rename(columns={df_reset.columns[0]: 'Date'}, inplace=True)
            date_col = 'Date'
    
    # Find price column
    price_col = _find_price_column(df_reset, 'close')
    if price_col is None:
        raise ValueError(f"Price column 'close' not found. Available: {df_reset.columns.tolist()}")
    
    result = pd.DataFrame({
        'ds': pd.to_datetime(df_reset[date_col]),
        'y': df_reset[price_col].values
    })
    
    result = result.dropna()
    
    print(f"   ✅ Prepared {len(result)} rows for Prophet")
    print(f"   Date range: {result['ds'].min()} to {result['ds'].max()}")
    
    return result

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data...")
    
    original_len = len(df)
    
    # Remove duplicates
    date_col = _find_date_column(df)
    if date_col:
        df = df[~df.duplicated(subset=[date_col], keep='first')].copy()
    
    # Handle missing values
    price_col = _find_price_column(df, 'close')
    if price_col:
        missing = df[price_col].isna().sum()
        if missing > 0:
            print(f"   Filling {missing} missing prices")
            df[price_col] = df[price_col].ffill().bfill()
    
    removed = original_len - len(df)
    if removed > 0:
        print(f"   Removed {removed} rows")
    
    print(f"   Final records: {len(df)}")
    
    return df
