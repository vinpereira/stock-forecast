import pandas as pd
import logging

logger = logging.getLogger(__name__)

def prepare_for_prophet(df):
    """Convert stock data to Prophet format."""
    logger.info("Preparing data for Prophet")
    
    # If MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get Date as column
    df_reset = df.reset_index()
    
    if 'Date' not in df_reset.columns:
        raise ValueError(
            f"Cannot find 'Date' column. Available: {df_reset.columns.tolist()}"
        )
    
    result = pd.DataFrame({
        'ds': pd.to_datetime(df_reset['Date']),
        'y': df_reset['Close'].values
    })
    
    result = result.dropna()
    logger.info(f"Prepared {len(result)} rows for Prophet")
    
    return result