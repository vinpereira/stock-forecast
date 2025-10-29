import pandas as pd
from src.data import prepare_for_prophet

def test_prepare_for_prophet():
    # Create sample data
    data = pd.DataFrame({
        'Close': [100, 101, 102],
        'Open': [99, 100, 101]
    })
    data.index = pd.date_range('2024-01-01', periods=3)
    data.index.name = 'Date'
    
    result = prepare_for_prophet(data)
    
    assert 'ds' in result.columns
    assert 'y' in result.columns
    assert len(result) == 3
