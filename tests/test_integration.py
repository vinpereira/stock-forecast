import pytest
import subprocess
from pathlib import Path

@pytest.mark.slow
def test_full_pipeline_runs():
    result = subprocess.run(
        ['uv', 'run', 'python', 'main.py', '--symbol', 'AAPL', '--days', '30'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    # Should complete successfully
    assert result.returncode == 0
    assert 'FORECASTING COMPLETED SUCCESSFULLY' in result.stdout


@pytest.mark.slow
def test_outputs_are_generated():
    # Run forecast
    subprocess.run(
        ['uv', 'run', 'python', 'main.py', '--symbol', 'AAPL', '--days', '30'],
        capture_output=True,
        timeout=120
    )
    
    # Check outputs
    output_dir = Path('./outputs')
    
    assert (output_dir / 'forecast_AAPL.png').exists()
    assert (output_dir / 'forecast_components_AAPL.png').exists()
    assert (output_dir / 'forecast_AAPL.csv').exists()

@pytest.mark.slow
@pytest.mark.parametrize("symbol", ['AAPL', 'GOOGL', 'MSFT'])
def test_pipeline_works_for_different_stocks(symbol):
    result = subprocess.run(
        ['uv', 'run', 'python', 'main.py', '--symbol', symbol, '--days', '30'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    assert result.returncode == 0
    assert 'FORECASTING COMPLETED SUCCESSFULLY' in result.stdout

@pytest.mark.slow
def test_pipeline_with_invalid_symbol():
    result = subprocess.run(
        ['uv', 'run', 'python', 'main.py', '--symbol', 'INVALID123', '--days', '30'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Should exit with error
    assert result.returncode == 1
    assert 'Invalid symbol' in result.stdout or 'Error' in result.stdout
