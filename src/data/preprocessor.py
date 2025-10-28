import pandas as pd

def prepare_for_prophet(df):
    df = df.reset_index()
    result = pd.DataFrame({
        'ds': df['Date'],
        'y': df['Close']
    })
    return result
