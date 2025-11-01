import pytest
import pandas as pd

@pytest.fixture
def sample_stock_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + pd.Series(range(len(dates))) * 0.1,
        'High': 102 + pd.Series(range(len(dates))) * 0.1,
        'Low': 98 + pd.Series(range(len(dates))) * 0.1,
        'Close': 100 + pd.Series(range(len(dates))) * 0.1,
        'Volume': 1000000
    })
    data.set_index('Date', inplace=True)
    return data

@pytest.fixture
def sample_prophet_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    return pd.DataFrame({
        'ds': dates,
        'y': 100 + pd.Series(range(len(dates))) * 0.1
    })

@pytest.fixture
def sample_forecast():
    dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
    return pd.DataFrame({
        'ds': dates,
        'yhat': 100 + pd.Series(range(len(dates))) * 0.1,
        'yhat_lower': 95 + pd.Series(range(len(dates))) * 0.1,
        'yhat_upper': 105 + pd.Series(range(len(dates))) * 0.1
    })
