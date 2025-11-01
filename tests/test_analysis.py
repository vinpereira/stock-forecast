import pytest
import pandas as pd
from src.analysis.forecast import ForecastAnalyzer

def test_analyzer_initialization():
    analyzer = ForecastAnalyzer()
    assert analyzer is not None

def test_get_future_values(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    future = analyzer.get_future_values(sample_forecast, days=30)
    
    assert isinstance(future, pd.DataFrame)
    assert len(future) <= 30

@pytest.mark.parametrize("days", [7, 30, 90, 365])
def test_get_future_values_different_periods(sample_forecast, days):
    analyzer = ForecastAnalyzer()
    
    future = analyzer.get_future_values(sample_forecast, days=days)
    
    assert isinstance(future, pd.DataFrame)
    assert len(future) <= days

def test_find_optimal_sell_date(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    result = analyzer.find_optimal_sell_date(sample_forecast)
    
    assert 'date' in result
    assert 'price' in result
    assert result['date'] is not None

def test_find_optimal_sell_date_with_date_range(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    start = sample_forecast['ds'].min().strftime('%Y-%m-%d')
    end = sample_forecast['ds'].max().strftime('%Y-%m-%d')
    
    result = analyzer.find_optimal_sell_date(sample_forecast, start, end)
    
    assert 'date' in result
    assert 'price' in result

def test_find_optimal_sell_date_no_data():
    analyzer = ForecastAnalyzer()
    
    # Empty forecast
    empty_forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_upper', 'yhat_lower'])
    
    result = analyzer.find_optimal_sell_date(empty_forecast)
    
    assert 'error' in result

def test_calculate_volatility(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    volatility = analyzer.calculate_volatility(sample_forecast, window=30)
    
    assert 'std_dev' in volatility
    assert 'avg_confidence_range' in volatility
    assert volatility['std_dev'] >= 0

@pytest.mark.parametrize("window", [7, 30, 90])
def test_calculate_volatility_different_windows(sample_forecast, window):
    analyzer = ForecastAnalyzer()
    
    volatility = analyzer.calculate_volatility(sample_forecast, window=window)
    
    assert 'std_dev' in volatility
    assert isinstance(volatility['std_dev'], float)

def test_generate_scenarios(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    scenarios = analyzer.generate_scenarios(sample_forecast)
    
    assert 'optimistic' in scenarios
    assert 'expected' in scenarios
    assert 'pessimistic' in scenarios
    
    # Optimistic should be higher than pessimistic
    assert scenarios['optimistic']['price'] > scenarios['pessimistic']['price']

def test_generate_scenarios_with_target_date(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    target = sample_forecast['ds'].max().strftime('%Y-%m-%d')
    scenarios = analyzer.generate_scenarios(sample_forecast, target_date=target)
    
    assert 'expected' in scenarios
    assert 'price' in scenarios['expected']

def test_export_to_csv(sample_forecast, tmp_path):
    analyzer = ForecastAnalyzer()
    
    output_file = tmp_path / "test_forecast.csv"
    
    analyzer.export_to_csv(sample_forecast, str(output_file), include_components=False)
    
    assert output_file.exists()
    
    # Read back and verify
    df = pd.read_csv(output_file)
    assert 'ds' in df.columns
    assert 'yhat' in df.columns

def test_export_to_csv_with_components(sample_forecast, tmp_path):
    analyzer = ForecastAnalyzer()
    
    output_file = tmp_path / "test_forecast_full.csv"
    
    analyzer.export_to_csv(sample_forecast, str(output_file), include_components=True)
    
    assert output_file.exists()
    
    # Should have all columns
    df = pd.read_csv(output_file)
    assert len(df.columns) >= 4  # At least ds, yhat, yhat_lower, yhat_upper
