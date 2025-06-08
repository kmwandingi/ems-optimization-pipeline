#!/usr/bin/env python
"""
Configuration Loader for EMS Optimization Pipeline
Loads centralized configuration from YAML and provides easy access to all parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigLoader:
    """
    Centralized configuration loader for EMS optimization pipeline.
    Loads default configuration and allows environment-specific overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default config/default.yaml
        """
        if config_path is None:
            # Find config/default.yaml relative to this file
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Common environment variables that might override config
        env_mappings = {
            'EMS_DEBUG': ('app', 'debug'),
            'EMS_ENVIRONMENT': ('app', 'environment'),
            'EMS_DATABASE_PATH': ('database', 'path'),
            'EMS_MLFLOW_URI': ('mlflow', 'tracking_uri'),
            'EMS_LOG_LEVEL': ('logging', 'level'),
            'EMS_MAX_ITERATIONS': ('optimization', 'global_optimizer', 'max_iterations'),
            'EMS_SOLVER_TIMEOUT': ('optimization', 'global_optimizer', 'solver_timeout'),
        }
        
        for env_var, path_tuple in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                
                # Handle nested paths
                if len(path_tuple) == 2:
                    section, key = path_tuple
                    if section in config:
                        config[section][key] = value
                elif len(path_tuple) == 3:
                    section, subsection, key = path_tuple
                    if section in config and subsection in config[section]:
                        config[section][subsection][key] = value
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'battery.default.capacity')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get('battery.default.capacity')  # Returns 10.0
            config.get('optimization.global_optimizer.max_iterations')  # Returns 5
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_battery_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get battery configuration for specified profile."""
        return self.get(f'battery.{profile}', {})
    
    def get_ev_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get EV configuration for specified profile."""
        return self.get(f'ev.{profile}', {})
    
    def get_grid_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get grid configuration for specified profile."""
        return self.get(f'grid.{profile}', {})
    
    def get_building_config(self, building_type: str = 'residential') -> Dict[str, Any]:
        """Get building configuration for specified type."""
        return self.get(f'building.{building_type}', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.get('optimization', {})
    
    def get_probability_model_config(self) -> Dict[str, Any]:
        """Get probability model configuration."""
        return self.get('probability_model', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get('mlflow', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.get('pipeline', {})
    
    def get_agents_config(self) -> Dict[str, Any]:
        """Get agents configuration."""
        return self.get('agents', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get('paths', {})
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('app.debug', False)
    
    def get_environment(self) -> str:
        """Get current environment."""
        return self.get('app.environment', 'development')
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields and constraints.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = [
            'app', 'database', 'paths', 'battery', 'ev', 'grid', 
            'optimization', 'mlflow', 'logging', 'pipeline', 'agents'
        ]
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required configuration section: {section}")
                return False
        
        # Validate critical constraints
        if not self.get('agents.enable_fallbacks', True):
            # This is correct - fallbacks should be disabled for production
            pass
        
        if not self.get('agents.require_24_hour_schedules', False):
            print("Warning: 24-hour schedule validation is disabled")
        
        if self.get('optimization.global_optimizer.max_iterations', 0) <= 0:
            print("Error: max_iterations must be positive")
            return False
        
        return True
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save config. If None, saves to original config_path
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Force reload of configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        New ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance


# Convenience functions for common configuration access
def get_battery_params(profile: str = 'default') -> Dict[str, Any]:
    """Get battery parameters for specified profile."""
    return get_config().get_battery_config(profile)


def get_ev_params(profile: str = 'default') -> Dict[str, Any]:
    """Get EV parameters for specified profile."""
    return get_config().get_ev_config(profile)


def get_grid_params(profile: str = 'default') -> Dict[str, Any]:
    """Get grid parameters for specified profile."""
    return get_config().get_grid_config(profile)


def get_optimization_params() -> Dict[str, Any]:
    """Get optimization parameters."""
    return get_config().get_optimization_config()


def get_mlflow_params() -> Dict[str, Any]:
    """Get MLflow parameters."""
    return get_config().get_mlflow_config()


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"Environment: {config.get_environment()}")
    print(f"Debug mode: {config.is_debug_mode()}")
    
    # Test various configuration access methods
    print(f"Battery capacity: {config.get('battery.default.capacity')}")
    print(f"Max iterations: {config.get('optimization.global_optimizer.max_iterations')}")
    print(f"MLflow URI: {config.get('mlflow.tracking_uri')}")
    
    # Test validation
    is_valid = config.validate_config()
    print(f"Configuration valid: {is_valid}")
    
    # Test profile-based access
    battery_config = config.get_battery_config('default')
    print(f"Battery config: {battery_config}")
    
    ev_config = config.get_ev_config('v2g_enabled')
    print(f"V2G EV config: {ev_config}")