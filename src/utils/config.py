import yaml

def load_config(path='config.yaml'):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    
    # Validate required fields
    required = ['stock', 'forecast']
    for field in required:
        if field not in cfg:
            raise ValueError(f"Missing required config: {field}")
    
    return cfg
