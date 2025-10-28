import yfinance as yf

class Fetcher:
    def fetch(self, symbol, start, end):
        data = yf.download(symbol, start=start, end=end)
        return data