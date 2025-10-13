"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison initialization module
===============================================================
"""

import json
from pathlib import Path
from typing import Dict, List


def load_config(instrument_type: str, config_dir: str = "./configs") -> Dict:
    """
    Load configuration for a specific instrument type.
    
    Parameters
    ----------
    instrument_type : str
        The instrument type (e.g., 'vaisala_ceilometer', 'hsrl', etc.)
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    Dict
        Configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If configuration file is not found
    ValueError
        If JSON file is invalid
    """
    config_file = Path(config_dir) / f"{instrument_type}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file}: {e}")


def get_variable_mapping(instrument_type: str, config_dir: str = "./configs") -> Dict[str, str]:
    """
    Get variable mapping for an instrument type.
    
    Parameters
    ----------
    instrument_type : str
        The instrument type
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping standard names to instrument-specific names
    """
    config = load_config(instrument_type, config_dir)
    return config["variables"]