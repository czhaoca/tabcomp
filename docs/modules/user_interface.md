# User Interface Module

## Overview

The User Interface module provides a graphical interface for file selection, configuration, and result visualization.

## Core Components

### 1. Main Window

```python
class MainWindow:
    """
    Main application window handling primary user interactions.
    """
    def initialize_ui(self) -> None:
        """
        Sets up main window layout and components.
        """

    def create_menu_bar(self) -> None:
        """
        Creates application menu with options.
        """

    async def handle_file_selection(self, file_type: str) -> str:
        """
        Handles file selection dialog and validation.
        """
```

### 2. File Selection Panel

```python
class FileSelectionPanel:
    """
    Handles file selection and initial validation.
    """
    def create_file_browsers(self) -> None:
        """
        Creates file browser interfaces for both files.
        """

    def validate_selection(self, file_path: str) -> ValidationResult:
        """
        Validates selected file before processing.
        """

    def handle_sheet_selection(self, excel_file: str) -> List[str]:
        """
        Manages Excel sheet selection interface.
        """
```

### 3. Field Mapping Interface

```python
class FieldMappingInterface:
    """
    Handles field mapping and primary key selection.
    """
    def display_fields(self,
                      fields1: List[str],
                      fields2: List[str]) -> None:
        """
        Shows field mapping interface with auto-matched suggestions.
        """

    def handle_primary_key_selection(self,
                                   fields: List[str]) -> List[str]:
        """
        Manages primary key selection interface.
        """
```

### 4. Progress Indicator

```python
class ProgressIndicator:
    """
    Manages progress visualization during operations.
    """
    def update_progress(self,
                       current: int,
                       total: int,
                       message: str) -> None:
        """
        Updates progress bar and status message.
        """

    def show_completion_dialog(self,
                             results: ComparisonResult) -> None:
        """
        Shows completion dialog with summary.
        """
```

## Event Handling

### User Actions

```python
class UIEventHandler:
    """
    Handles UI event processing.
    """
    def register_handlers(self) -> None:
        """
        Registers event handlers for UI components.
        """

    async def handle_comparison_start(self) -> None:
        """
        Initiates comparison process with loading indicator.
        """
```

## Interface Components

### Dialog Boxes

```python
class DialogManager:
    """
    Manages application dialogs.
    """
    def show_error_dialog(self,
                         message: str,
                         details: str) -> None:
        """
        Displays error dialog with details.
        """

    def show_warning_dialog(self,
                          message: str,
                          actions: List[str]) -> str:
        """
        Shows warning dialog with action choices.
        """
```

## Accessibility Features

### Keyboard Navigation

```python
class KeyboardNavigator:
    """
    Handles keyboard navigation and shortcuts.
    """
    def setup_shortcuts(self) -> None:
        """
        Sets up keyboard shortcuts.
        """

    def handle_tab_navigation(self, event: Event) -> None:
        """
        Manages tab-based navigation.
        """
```

## Theme Management

### Color Schemes

```python
@dataclass
class UITheme:
    primary_color: str
    secondary_color: str
    accent_color: str
    text_color: str
```

### Theme Application

```python
class ThemeManager:
    """
    Manages UI theme application.
    """
    def apply_theme(self, theme: UITheme) -> None:
        """
        Applies theme to all UI components.
        """
```

## Examples

### Basic Window Setup

```python
app = TabCompUI()
app.initialize_ui()
app.show()
```

### Custom Theme Application

```python
theme = UITheme(
    primary_color="#2C3E50",
    secondary_color="#ECF0F1",
    accent_color="#3498DB",
    text_color="#2C3E50"
)
app.theme_manager.apply_theme(theme)
```
