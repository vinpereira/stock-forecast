import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}"
            )
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Configuration file is empty")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_stock_config(self) -> Dict[str, Any]:
        return self.get('stock', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.get('model', {})
    
    def get_forecast_config(self) -> Dict[str, Any]:
        return self.get('forecast', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        return self.get('output', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        return self.get('visualization', {})
    
    def validate(self) -> bool:
        required = {
            'stock': ['symbol', 'start', 'end'],
            'forecast': ['days'],
        }
        
        for section, fields in required.items():
            section_config = self.get(section)
            if not section_config:
                raise ValueError(f"Missing section: {section}")
            
            for field in fields:
                if field not in section_config:
                    raise ValueError(f"Missing field: {section}.{field}")
        
        return True

def load_config(config_file: str = "config.yaml") -> Config:
    return Config(config_file)
