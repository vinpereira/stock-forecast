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


def test_find_optimal_sell_date(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    result = analyzer.find_optimal_sell_date(sample_forecast)
    
    assert 'date' in result
    assert 'price' in result
    assert result['date'] is not None


def test_calculate_volatility(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    volatility = analyzer.calculate_volatility(sample_forecast, window=30)
    
    assert 'std_dev' in volatility
    assert 'avg_confidence_range' in volatility
    assert volatility['std_dev'] >= 0


def test_generate_scenarios(sample_forecast):
    analyzer = ForecastAnalyzer()
    
    scenarios = analyzer.generate_scenarios(sample_forecast)
    
    assert 'optimistic' in scenarios
    assert 'expected' in scenarios
    assert 'pessimistic' in scenarios
