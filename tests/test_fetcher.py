from src.data import Fetcher

def test_fetcher_exists():
    f = Fetcher()
    assert f is not None

def test_fetcher_has_fetch_method():
    f = Fetcher()
    assert hasattr(f, 'fetch')
