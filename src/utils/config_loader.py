import yaml
from typing import Dict 

def load_config(config_path: str) -> Dict:
    """
    Return the Config Dictionary
    
    Args:
        config_path: Path to the Config Parameter File
    """

    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    return config