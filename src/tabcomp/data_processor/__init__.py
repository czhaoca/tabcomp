"""
TabComp Data Processor Module.

This module provides functionality for comparing and analyzing tabular data,
with support for data normalization, field matching, and detailed comparison
operations.

Example:
    >>> from tabcomp.data_processor import DataProcessor
    >>> processor = DataProcessor()
    >>> result = processor.compare_tables(table1, table2)
"""

from typing import Dict, List, Optional, Union, Any, TypeVar, Callable
from dataclasses import dataclass, field
import pandas as pd
import logging
from datetime import datetime
from functools import wraps
import threading
import warnings

# Type variables for generic methods
DF = TypeVar("DF", bound=pd.DataFrame)

# Version information
__version__ = "1.0.0"
__author__ = "TabComp Team"
__status__ = "Production"


@dataclass
class ComparisonOptions:
    """Configuration options for table comparison."""

    # Performance options
    chunk_size: int = field(default=10000)
    max_memory_mb: int = field(default=500)
    parallel_threshold: int = field(default=100_000)

    # Comparison behavior
    case_sensitive: bool = field(default=False)
    ignore_whitespace: bool = field(default=True)
    date_format: str = field(default="%Y-%m-%d")
    numeric_precision: int = field(default=8)
    match_threshold: float = field(default=0.85)

    # Optional configurations
    ignore_columns: List[str] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    column_aliases: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if not 0 <= self.match_threshold <= 1:
            raise ValueError("match_threshold must be between 0 and 1")
        # Validate date format
        try:
            datetime.now().strftime(self.date_format)
        except ValueError as e:
            raise ValueError(f"Invalid date_format: {str(e)}")


@dataclass
class ComparisonResult:
    """Results of table comparison operation."""

    differences: pd.DataFrame
    summary: Dict[str, Any]
    total_rows: int
    total_differences: int
    execution_time: float
    memory_usage: int
    comparison_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def difference_rate(self) -> float:
        """Calculate percentage of rows with differences."""
        return (
            (self.total_differences / self.total_rows * 100)
            if self.total_rows > 0
            else 0
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get formatted summary of comparison results."""
        return {
            **self.summary,
            "total_rows": self.total_rows,
            "total_differences": self.total_differences,
            "difference_rate": self.difference_rate,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "timestamp": self.comparison_timestamp,
        }


class DataProcessorError(Exception):
    """Base exception for data processor module."""

    pass


def validate_input(func: Callable) -> Callable:
    """Decorator for validating method inputs."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], pd.DataFrame):
            df = args[0]
            if len(df) == 0:
                raise ValueError("Empty DataFrame provided")
            if self.options.required_columns:
                missing = set(self.options.required_columns) - set(df.columns)
                if missing:
                    raise ValueError(f"Required columns missing: {missing}")
        return func(self, *args, **kwargs)

    return wrapper


class DataProcessor:
    """
    Main interface for data processing and comparison operations.

    This class orchestrates the data processing workflow, managing the
    interaction between different components for normalization, matching,
    and comparison operations.
    """

    def __init__(
        self,
        options: Optional[ComparisonOptions] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataProcessor with optional configuration.

        Args:
            options: Configuration options for comparison operations.
                    If not provided, uses default values.
            logger: Optional logger instance for detailed logging.
        """
        self.options = options or ComparisonOptions()
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

        # Initialize component trackers
        self._components_initialized = False
        self._memory_warning_shown = False

        # Defer component initialization until needed
        self._normalizer = None
        self._matcher = None
        self._comparator = None

    def _init_components(self) -> None:
        """Initialize components lazily when first needed."""
        if not self._components_initialized:
            with self._lock:
                if not self._components_initialized:
                    from .normalizer import DataNormalizer
                    from .matcher import FieldMatcher
                    from .comparator import TableComparator

                    self._normalizer = DataNormalizer()
                    self._matcher = FieldMatcher()
                    self._comparator = TableComparator()
                    self._components_initialized = True

    @validate_input
    def normalize_table(
        self,
        df: DF,
        date_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DF:
        """
        Normalize table data for consistent comparison.

        Args:
            df: Input DataFrame to normalize
            date_columns: Optional list of columns containing dates
            numeric_columns: Optional list of columns containing numeric values
            **kwargs: Additional keyword arguments for normalization

        Returns:
            Normalized DataFrame with standardized data types and formats

        Raises:
            ValueError: If column specifications are invalid
            TypeError: If data cannot be converted to specified types
        """
        self._init_components()
        self._check_memory_usage(df)

        try:
            return self._normalizer.normalize(
                df,
                date_columns=date_columns,
                numeric_columns=numeric_columns,
                date_format=self.options.date_format,
                numeric_precision=self.options.numeric_precision,
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}")
            raise DataProcessorError(f"Failed to normalize table: {str(e)}") from e

    @validate_input
    def match_fields(
        self,
        fields1: List[str],
        fields2: List[str],
        primary_key: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Match fields between two tables using intelligent matching algorithms.

        Args:
            fields1: List of field names from first table
            fields2: List of field names from second table
            primary_key: Optional field(s) to use as primary key(s)
            **kwargs: Additional keyword arguments for matching

        Returns:
            Dictionary containing field mappings and match confidence scores

        Raises:
            ValueError: If primary key is invalid or not found in both tables
        """
        self._init_components()

        try:
            return self._matcher.match_fields(
                fields1,
                fields2,
                primary_key=primary_key,
                case_sensitive=self.options.case_sensitive,
                ignore_whitespace=self.options.ignore_whitespace,
                match_threshold=self.options.match_threshold,
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"Field matching failed: {str(e)}")
            raise DataProcessorError(f"Failed to match fields: {str(e)}") from e

    @validate_input
    def compare_tables(
        self,
        table1: DF,
        table2: DF,
        field_mapping: Optional[Dict[str, str]] = None,
        primary_key: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ComparisonResult:
        """
        Compare two tables and identify differences.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Optional manual field mapping override
            primary_key: Field(s) to use as primary key(s)
            **kwargs: Additional keyword arguments for comparison

        Returns:
            ComparisonResult containing detailed comparison results

        Raises:
            ValueError: If tables cannot be compared due to structural issues
            KeyError: If specified fields are not found in tables
        """
        self._init_components()
        self._check_memory_usage(table1, table2)

        start_time = datetime.now()
        self.logger.info("Starting table comparison")

        try:
            # Normalize tables
            table1_norm = self.normalize_table(table1)
            table2_norm = self.normalize_table(table2)

            # Match fields if mapping not provided
            if not field_mapping:
                match_result = self.match_fields(
                    list(table1_norm.columns), list(table2_norm.columns), primary_key
                )
                field_mapping = match_result["matched_fields"]

            # Perform comparison
            result = self._comparator.compare(
                table1_norm,
                table2_norm,
                field_mapping,
                primary_key,
                chunk_size=self.options.chunk_size,
                ignore_columns=self.options.ignore_columns,
                **kwargs,
            )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Comparison completed in {execution_time:.2f} seconds")

            return result

        except Exception as e:
            self.logger.error(f"Table comparison failed: {str(e)}")
            raise DataProcessorError(f"Failed to compare tables: {str(e)}") from e

    def _check_memory_usage(self, *dfs: pd.DataFrame) -> None:
        """Check if DataFrame sizes might cause memory issues."""
        total_memory_mb = sum(
            df.memory_usage(deep=True).sum() / (1024 * 1024) for df in dfs
        )
        if (
            total_memory_mb > self.options.max_memory_mb
            and not self._memory_warning_shown
        ):
            warnings.warn(
                f"Input data size ({total_memory_mb:.1f}MB) exceeds configured limit "
                f"({self.options.max_memory_mb}MB). This may cause performance issues.",
                ResourceWarning,
            )
            self._memory_warning_shown = True


# Convenience functions
def normalize_table(df: DF, **kwargs: Any) -> DF:
    """Convenience function for table normalization."""
    processor = DataProcessor()
    return processor.normalize_table(df, **kwargs)


def match_fields(
    fields1: List[str], fields2: List[str], **kwargs: Any
) -> Dict[str, Any]:
    """Convenience function for field matching."""
    processor = DataProcessor()
    return processor.match_fields(fields1, fields2, **kwargs)


def compare_tables(table1: DF, table2: DF, **kwargs: Any) -> ComparisonResult:
    """Convenience function for table comparison."""
    processor = DataProcessor()
    return processor.compare_tables(table1, table2, **kwargs)
