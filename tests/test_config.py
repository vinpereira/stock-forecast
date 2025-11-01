import pytest
from src.utils.config import Config


def test_load_config():
    config = Config('config.yaml')
    assert config.config is not None


def test_config_has_required_sections():
    config = Config('config.yaml')
    
    assert 'stock' in config.config
    assert 'forecast' in config.config
    assert 'output' in config.config


@pytest.mark.parametrize("section,expected_keys", [
    ('stock', ['symbol', 'start']),
    ('forecast', ['days']),
    ('output', ['directory']),
])
def test_config_sections_have_required_keys(section, expected_keys):
    config = Config('config.yaml')
    section_config = config.get(section, {})
    
    for key in expected_keys:
        assert key in section_config, f"Missing key '{key}' in section '{section}'"


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


def test_config_missing_file():
    with pytest.raises(FileNotFoundError):
        Config('nonexistent.yaml')


def test_config_get_with_default():
    config = Config('config.yaml')
    
    # Non-existent key should return default
    assert config.get('nonexistent.key', 'default_value') == 'default_value'
