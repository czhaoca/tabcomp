# Output Generator Module

## Overview

The Output Generator module handles report generation, formatting, and logging for comparison results.

## Core Components

### 1. Report Generator

```python
class ReportGenerator:
    """
    Generates Excel reports with comparison results.
    """
    def generate_report(self,
                       results: ComparisonResult,
                       output_path: str) -> None:
        """
        Creates formatted Excel report.
        """

    def create_summary(self,
                      results: ComparisonResult) -> pd.DataFrame:
        """
        Generates summary statistics.
        """
```

### 2. Excel Formatter

```python
class ExcelFormatter:
    """
    Handles Excel formatting and styling.
    """
    def apply_formatting(self,
                        writer: pd.ExcelWriter,
                        sheet_name: str) -> None:
        """
        Applies standard formatting to Excel sheet.
        """

    def highlight_differences(self,
                            writer: pd.ExcelWriter,
                            diff_cells: List[str]) -> None:
        """
        Highlights cells with differences.
        """
```

### 3. Logger

```python
class Logger:
    """
    Handles logging operations.
    """
    def log_operation(self,
                     message: str,
                     level: LogLevel) -> None:
        """
        Logs operation with timestamp.
        """

    def create_log_file(self,
                       output_dir: str) -> str:
        """
        Creates log file in output directory.
        """
```

## Report Structure

### Summary Sheet

```python
@dataclass
class SummaryContent:
    execution_time: datetime
    duration: float
    errors: List[str]
    file_stats: Dict[str, FileStats]
    comparison_stats: ComparisonStats
```

### Difference Sheet

```python
@dataclass
class DifferenceSheetFormat:
    frozen_columns: List[str]
    highlight_colors: Dict[str, str]
    column_order: List[str]
```

## Formatting Rules

### Cell Formatting

```python
class CellFormatter:
    """
    Handles cell-level formatting.
    """
    def format_header(self, cell: str) -> Dict:
        """
        Returns header cell formatting.
        """

    def format_difference(self, cell: str) -> Dict:
        """
        Returns difference cell formatting.
        """
```

### Color Schemes

```python
DEFAULT_COLORS = {
    'file1': '#ADD8E6',  # Light blue
    'file2': '#90EE90',  # Light green
    'diff': '#FFFF00'    # Yellow
}
```

## Logging Standards

### Log Format

```python
@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any]
```

### Log Levels

```python
class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
```

## Examples

### Basic Report Generation

```python
generator = OutputGenerator()
generator.generate_report(results, "output.xlsx")
```

### Custom Formatted Report

```python
options = ReportOptions(
    colors=custom_colors,
    freeze_columns=['file', 'row', 'id'],
    bold_headers=True
)
generator.generate_report_with_options(results, "output.xlsx", options)
```
