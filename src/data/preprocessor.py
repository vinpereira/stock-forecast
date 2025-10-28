import pandas as pd

def prepare_for_prophet(df):
    df = df.reset_index()
    
    result = pd.DataFrame({
        'ds': pd.to_datetime(df['Date']),
        'y': df['Close'].values
    })
    
    result = result.dropna()
    return result