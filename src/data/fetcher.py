import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Fetcher:
    def fetch(self, symbol, start, end):
        print(f"📥 Downloading {symbol} data from Yahoo Finance...")
        print(f"   Period: {start} to {end}")
        
        data = yf.download(
            symbol, 
            start=start, 
            end=end, 
            auto_adjust=True,
            progress=False
        )
        
        if data is None or data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        print(f"✅ Downloaded {len(data)} rows")
        
        if len(data) > 0:
            last_close = data['close'].iloc[-1]
            last_date = data.index[-1]
            print(f"   Last closing price: ${last_close:.2f} on {last_date}")
        
        return data
