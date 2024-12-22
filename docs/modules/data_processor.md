# Data Processor Module

## Overview

The Data Processor module handles data normalization, field matching, and comparison operations between tables.

## Core Components

### 1. Data Normalizer

```python
class DataNormalizer:
    """
    Normalizes data for consistent comparison.
    """
    def normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts dates to YYYY-MM-DD format.
        Invalid dates are replaced with 'ERROR'.
        """

    def normalize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes string values (trim, case, special chars).
        """

    def normalize_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes numeric formats.
        """
```

### 2. Field Matcher

```python
class FieldMatcher:
    """
    Handles field matching between tables.
    """
    def match_fields(self,
                    fields1: List[str],
                    fields2: List[str]) -> FieldMapping:
        """
        Matches fields between tables using similarity algorithms.
        """

    def validate_primary_keys(self,
                            df: pd.DataFrame,
                            keys: List[str]) -> ValidationResult:
        """
        Validates primary key uniqueness and completeness.
        """
```

### 3. Comparison Engine

```python
class ComparisonEngine:
    """
    Performs table comparison operations.
    """
    def compare_tables(self,
                      table1: pd.DataFrame,
                      table2: pd.DataFrame,
                      mapping: FieldMapping) -> ComparisonResult:
        """
        Compares tables and identifies differences.
        """

    def analyze_differences(self,
                          diff_df: pd.DataFrame) -> DifferenceAnalysis:
        """
        Analyzes and categorizes differences.
        """
```

## Data Type Handling

### Date Processing

```python
class DateHandler:
    """
    Handles date format standardization.
    """
    DEFAULT_FORMAT = "%Y-%m-%d"
    ERROR_VALUE = "ERROR"

    def standardize_date(self, value: Any) -> str:
        """
        Converts dates to standard format or ERROR.
        """
```

### Numeric Processing

```python
class NumericHandler:
    """
    Handles numeric value standardization.
    """
    def standardize_numeric(self, value: Any) -> float:
        """
        Converts numeric strings to standard format.
        """
```

## Performance Optimization

### Parallel Processing

```python
class ParallelProcessor:
    """
    Handles parallel comparison operations.
    """
    def parallel_compare(self,
                        chunks1: List[pd.DataFrame],
                        chunks2: List[pd.DataFrame]) -> ComparisonResult:
        """
        Performs comparison in parallel.
        """
```

### Memory Management

```python
class MemoryOptimizer:
    """
    Optimizes memory usage during comparison.
    """
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes DataFrame memory usage.
        """
```

## Examples

### Basic Comparison

```python
processor = DataProcessor()
result = processor.compare_tables(df1, df2, primary_keys=['id'])
```

### Advanced Comparison

```python
options = ComparisonOptions(
    case_sensitive=False,
    ignore_whitespace=True,
    date_format="%Y-%m-%d"
)
result = processor.compare_with_options(df1, df2, options)
```
