import subprocess
from pathlib import Path


def test_full_pipeline_runs():
    import subprocess
    
    result = subprocess.run(
        ['uv', 'run', 'python', 'main.py', '--symbol', 'AAPL', '--days', '30'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    # Should complete successfully
    assert result.returncode == 0
    assert 'FORECASTING COMPLETED SUCCESSFULLY' in result.stdout


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
