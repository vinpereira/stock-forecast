from src.utils.config import Config


def test_load_config():
    config = Config('config.yaml')
    assert config.config is not None


def test_config_has_required_sections():
    config = Config('config.yaml')
    
    assert 'stock' in config.config
    assert 'forecast' in config.config
    assert 'output' in config.config


def test_config_getters():
    config = Config('config.yaml')
    
    stock_config = config.get_stock_config()
    assert 'symbol' in stock_config
    assert 'start' in stock_config
    
    forecast_config = config.get_forecast_config()
    assert 'days' in forecast_config


def test_config_validation():
    config = Config('config.yaml')
    
    # Should not raise any exceptions
    config.validate()
