import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Fetcher:
    def fetch(self, symbol, start, end):
        logger.info(f"Fetching {symbol} from {start} to {end}")
        
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
        
        logger.info(f"Downloaded {len(data)} rows")
        return data
