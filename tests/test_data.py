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
