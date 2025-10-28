import pandas as pd

def calculate_metrics(data):
    """Calculate basic stock metrics."""
    metrics = {
        'mean': data['Close'].mean(),
        'std': data['Close'].std(),
        'min': data['Close'].min(),
        'max': data['Close'].max(),
        'range': data['Close'].max() - data['Close'].min()
    }
    return metrics
