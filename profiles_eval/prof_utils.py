"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison utility functions
===============================================================
"""

from typing import Dict, List, Optional
import prof_init as pi


def get_common_variables(instrument_types: List[str], config_dir: str = "./configs") -> List[str]:
    """
    Get variables that are common across multiple instrument types.
    
    Parameters
    ----------
    instrument_types : List[str]
        List of instrument types to compare
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    List[str]
        List of common variable names (standard names)
    """
    if not instrument_types:
        return []
        
    # Load all configs and get variable sets
    variable_sets = []
    for inst_type in instrument_types:
        config = pi.load_config(inst_type, config_dir)
        variable_sets.append(set(config["variables"].keys()))
    
    # Find intersection
    common_vars = variable_sets[0]
    for var_set in variable_sets[1:]:
        common_vars &= var_set
        
    return list(common_vars)