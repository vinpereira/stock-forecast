import yfinance as yf
import pandas as pd

class Fetcher:
    def fetch(self, symbol, start, end):
        data = yf.download(
            symbol, 
            start=start, 
            end=end, 
            auto_adjust=True,
            progress=False
        )
        
        # Validate data was returned
        if data is None or data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Remove MultiIndex if present (single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data