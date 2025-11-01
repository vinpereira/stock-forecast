import pytest
import pandas as pd
from src.data.preprocessor import prepare_for_prophet


def test_prepare_for_prophet(sample_stock_data):
    result = prepare_for_prophet(sample_stock_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'ds' in result.columns
    assert 'y' in result.columns
    assert len(result) > 0


def test_prepare_for_prophet_has_correct_types(sample_stock_data):
    result = prepare_for_prophet(sample_stock_data)
    
    assert pd.api.types.is_datetime64_any_dtype(result['ds'])
    assert pd.api.types.is_numeric_dtype(result['y'])


def test_prepare_for_prophet_no_nulls(sample_stock_data):
    result = prepare_for_prophet(sample_stock_data)
    
    assert result['ds'].isna().sum() == 0
    assert result['y'].isna().sum() == 0


def test_prepare_for_prophet_preserves_data_count(sample_stock_data):
    result = prepare_for_prophet(sample_stock_data)
    
    # Should have same or fewer rows (if any NaNs were dropped)
    assert len(result) <= len(sample_stock_data)


@pytest.mark.parametrize("price_column", ['close', 'Close', 'CLOSE'])
def test_prepare_handles_different_column_names(sample_stock_data, price_column):
    # Rename column
    data = sample_stock_data.copy()
    if 'close' in data.columns:
        data = data.rename(columns={'close': price_column})
    else:
        data = data.rename(columns={'Close': price_column})
    
    result = prepare_for_prophet(data)
    
    assert 'ds' in result.columns
    assert 'y' in result.columns


def test_prepare_with_empty_dataframe():
    empty_df = pd.DataFrame()
    
    with pytest.raises((ValueError, KeyError)):
        prepare_for_prophet(empty_df)
