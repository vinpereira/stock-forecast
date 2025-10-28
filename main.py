from src.data.fetcher import Fetcher
from src.data.preprocessor import prepare_for_prophet
from src.utils.config import load_config

def main():
    cfg = load_config()
    
    symbol = cfg['stock']['symbol']
    start = cfg['stock']['start']
    end = cfg['stock']['end']
    
    print(f"Fetching {symbol}...")
    f = Fetcher()
    data = f.fetch(symbol, start, end)
    print(f"Got {len(data)} rows")
    
    prophet_data = prepare_for_prophet(data)
    print(prophet_data.head())

if __name__ == '__main__':
    main()