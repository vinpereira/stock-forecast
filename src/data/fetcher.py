import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Fetcher:
    def validate_symbol(self, symbol: str) -> bool:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'regularMarketPrice' in info or 'currentPrice' in info:
                return True
            return False
        except Exception:
            return False
    
    def get_stock_info(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            print(f"⚠️  Could not fetch stock info: {e}")
            return {}
    
    def fetch(self, symbol, start, end):
        print(f"📥 Downloading {symbol} data from Yahoo Finance...")
        
        # Handle 'today' keyword
        if end.lower() == 'today':
            end = datetime.now().strftime('%Y-%m-%d')
        
        # Check if end date is in future
        end_date_obj = datetime.strptime(end, '%Y-%m-%d')
        today_obj = datetime.now()
        
        if end_date_obj > today_obj:
            end = today_obj.strftime('%Y-%m-%d')
            print(f"   ⚠️  End date in future, using today: {end}")
        
        print(f"   Period: {start} to {end}")
        
        # yfinance end date is exclusive, add 1 day
        end_inclusive = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = yf.download(
            symbol, 
            start=start, 
            end=end_inclusive, 
            auto_adjust=True,
            progress=False
        )
        
        if data is None or data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Normalize column names to lowercase
        data.columns = data.columns.str.lower()
        
        print(f"✅ Downloaded {len(data)} rows")
        
        if len(data) > 0:
            last_close = data['close'].iloc[-1]
            last_date = data.index[-1]
            print(f"   Last closing price: ${last_close:.2f} on {last_date}")
        
        return data
