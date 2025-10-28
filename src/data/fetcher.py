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
        
        # Remove MultiIndex if present (single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data