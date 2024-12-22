# Configuration Manager Module

## Overview

The Configuration Manager module handles application settings, user preferences, and runtime configurations.

## Core Components

### 1. Settings Manager

```python
class SettingsManager:
    """
    Manages application settings and configurations.
    """
    def load_settings(self, config_path: str) -> Dict[str, Any]:
        """
        Loads settings from configuration file.
        """

    def save_settings(self,
                     settings: Dict[str, Any],
                     config_path: str) -> None:
        """
        Saves current settings to configuration file.
        """

    def validate_settings(self,
                         settings: Dict[str, Any]) -> ValidationResult:
        """
        Validates configuration values.
        """
```

### 2. Environment Manager

```python
class EnvironmentManager:
    """
    Handles environment-specific configurations.
    """
    def get_environment(self) -> str:
        """
        Returns current environment (dev/prod).
        """

    def load_env_config(self, env: str) -> Dict[str, Any]:
        """
        Loads environment-specific configuration.
        """
```

### 3. Runtime Configuration

```python
class RuntimeConfig:
    """
    Manages runtime configuration settings.
    """
    def update_config(self,
                     key: str,
                     value: Any) -> None:
        """
        Updates configuration during runtime.
        """

    def reset_to_defaults(self) -> None:
        """
        Resets configuration to default values.
        """
```

## Configuration Schema

### Application Settings

```python
@dataclass
class AppConfig:
    max_file_size: int = 500_000_000  # 500MB
    chunk_size: int = 10000
    timeout: int = 300  # seconds
    temp_dir: str = "./temp"
```

### Comparison Settings

```python
@dataclass
class ComparisonConfig:
    case_sensitive: bool = False
    ignore_whitespace: bool = True
    date_format: str = "%Y-%m-%d"
    numeric_precision: int = 8
```

### UI Settings

```python
@dataclass
class UIConfig:
    theme: str = "default"
    highlight_colors: Dict[str, str] = field(default_factory=dict)
    window_size: Tuple[int, int] = (1024, 768)
```

## Configuration File Format

### Default Configuration

```yaml
# config.yaml
app:
  max_file_size: 500000000
  chunk_size: 10000
  timeout: 300
  temp_dir: "./temp"

comparison:
  case_sensitive: false
  ignore_whitespace: true
  date_format: "%Y-%m-%d"
  numeric_precision: 8

ui:
  theme: "default"
  highlight_colors:
    file1: "#ADD8E6"
    file2: "#90EE90"
    diff: "#FFFF00"
  window_size: [1024, 768]
```

## Environment Handling

### Environment Detection

```python
class EnvironmentDetector:
    """
    Detects and configures environment-specific settings.
    """
    def detect_environment(self) -> str:
        """
        Detects current environment.
        """

    def load_env_variables(self, env: str) -> Dict[str, str]:
        """
        Loads environment variables.
        """
```

## Configuration Validation

### Validation Rules

```python
class ConfigValidator:
    """
    Validates configuration values.
    """
    def validate_file_size(self, size: int) -> bool:
        """
        Validates file size limits.
        """

    def validate_color_codes(self, colors: Dict[str, str]) -> bool:
        """
        Validates color code format.
        """
```

## Examples

### Basic Configuration Loading

```python
config_manager = ConfigManager()
config = config_manager.load_settings("config.yaml")
```

### Runtime Configuration Update

```python
runtime_config = RuntimeConfig()
runtime_config.update_config("max_file_size", 1000000000)
```

### Environment-Specific Configuration

```python
env_manager = EnvironmentManager()
env_config = env_manager.load_env_config("production")
```
