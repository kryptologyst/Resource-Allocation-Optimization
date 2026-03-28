"""
Configuration management utilities.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml
from omegaconf import OmegaConf


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional configuration parameters
        """
        self._config = {}
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._config.update(file_config)
        
        # Update with kwargs
        self._config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
