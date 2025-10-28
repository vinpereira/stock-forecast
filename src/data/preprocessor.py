import pandas as pd

def prepare_for_prophet(df):
    # If MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get Date as column
    df_reset = df.reset_index()
    
    result = pd.DataFrame({
        'ds': pd.to_datetime(df_reset['Date']),
        'y': df_reset['Close'].values
    })
    
    result = result.dropna()
    return result