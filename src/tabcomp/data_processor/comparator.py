"""
Table comparison module for TabComp.

This module handles the orchestration of comparison operations between tables,
delegating to specialized components for specific comparison tasks.

Example:
    >>> comparator = TableComparator()
    >>> result = comparator.compare(df1, df2, field_mapping, primary_key='id')
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import logging

from .comparison_strategies import FullTableStrategy, ChunkedStrategy
from .difference_handlers import DifferenceHandler
from .value_comparators import ValueComparator
from .exceptions import ComparisonError


@dataclass
class ComparisonConfig:
    """Configuration for table comparison operations."""

    chunk_size: int = 10000
    exact_match_only: bool = False
    ignore_case: bool = True
    ignore_whitespace: bool = True
    float_precision: int = 8
    date_match_precision: str = "second"  # "day", "hour", "minute", "second"
    null_equality: bool = True  # True: NULL == NULL, False: NULL != NULL


@dataclass
class ComparisonResult:
    """Results of table comparison operation."""

    differences: pd.DataFrame
    summary: Dict[str, Any]
    execution_time: float
    memory_usage: int
    total_rows: int
    total_differences: int
    comparison_timestamp: datetime


class TableComparator:
    """
    Handles table comparison operations.

    This class orchestrates the comparison process, utilizing specialized
    components for different aspects of the comparison operation.
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """
        Initialize the TableComparator with optional configuration.

        Args:
            config: Configuration options for comparison operations.
                   If not provided, uses default values.
        """
        self.config = config or ComparisonConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize specialized components
        self.value_comparator = ValueComparator(self.config)
        self.difference_handler = DifferenceHandler(self.value_comparator)
        self.full_strategy = FullTableStrategy(self.difference_handler)
        self.chunked_strategy = ChunkedStrategy(
            self.difference_handler, chunk_size=self.config.chunk_size
        )

    def compare(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_key: Union[str, List[str]],
        **kwargs,
    ) -> ComparisonResult:
        """
        Compare two tables and identify differences.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_key: Field(s) to use as primary key(s)
            **kwargs: Additional keyword arguments to override config

        Returns:
            ComparisonResult containing detailed comparison information

        Raises:
            ComparisonError: If comparison cannot be performed
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            self._validate_inputs(table1, table2, field_mapping, primary_key)

            # Update config with any provided overrides
            config = self._update_config(kwargs)

            # Standardize primary key to list
            if isinstance(primary_key, str):
                primary_key = [primary_key]

            # Initialize timing and results
            start_time = datetime.now()

            # Choose comparison strategy based on table size
            if len(table1) > config.chunk_size or len(table2) > config.chunk_size:
                differences = self.chunked_strategy.compare(
                    table1, table2, field_mapping, primary_key
                )
            else:
                differences = self.full_strategy.compare(
                    table1, table2, field_mapping, primary_key
                )

            # Create result
            result = ComparisonResult(
                differences=self._create_difference_dataframe(differences),
                summary=self._create_summary(differences),
                execution_time=(datetime.now() - start_time).total_seconds(),
                total_rows=len(table1),
                total_differences=len(differences),
                memory_usage=self._get_memory_usage([table1, table2]),
            )

            return result

        except Exception as e:
            raise ComparisonError(f"Failed to compare tables: {str(e)}")

    def _validate_inputs(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_key: Union[str, List[str]],
    ) -> None:
        """
        Validate comparison inputs.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Field mapping dictionary
            primary_key: Primary key field(s)

        Raises:
            ValueError: If inputs are invalid
        """
        if len(table1) == 0 or len(table2) == 0:
            raise ValueError("Empty tables cannot be compared")

        # Convert primary_key to list for validation
        primary_keys = [primary_key] if isinstance(primary_key, str) else primary_key

        # Validate primary keys exist in tables
        for pk in primary_keys:
            if pk not in table1.columns:
                raise ValueError(f"Primary key '{pk}' not found in first table")
            if field_mapping.get(pk, pk) not in table2.columns:
                raise ValueError(f"Primary key '{pk}' not found in second table")

        # Validate field mapping
        for field1, field2 in field_mapping.items():
            if field1 not in table1.columns:
                raise ValueError(f"Field '{field1}' not found in first table")
            if field2 not in table2.columns:
                raise ValueError(f"Field '{field2}' not found in second table")

    def _create_difference_dataframe(
        self, differences: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert difference records to DataFrame.

        Args:
            differences: List of difference records

        Returns:
            DataFrame containing formatted difference information
        """
        if not differences:
            return pd.DataFrame()

        return pd.DataFrame(differences).sort_values(
            by=["difference_type", "field_name"]
        )

    def _create_summary(self, differences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics from difference records.

        Args:
            differences: List of difference records

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "total_differences": len(differences),
            "by_type": {"missing": 0, "added": 0, "changed": 0},
            "by_field": {},
        }

        for diff in differences:
            diff_type = diff["difference_type"]
            field_name = diff["field_name"]

            summary["by_type"][diff_type] += 1
            summary["by_field"][field_name] = summary["by_field"].get(field_name, 0) + 1

        return summary

    def _get_memory_usage(self, dataframes: List[pd.DataFrame]) -> int:
        """
        Calculate total memory usage of DataFrames.

        Args:
            dataframes: List of DataFrames to measure

        Returns:
            Total memory usage in bytes
        """
        return sum(df.memory_usage(deep=True).sum() for df in dataframes)

    def _update_config(self, kwargs: Dict[str, Any]) -> ComparisonConfig:
        """
        Update configuration with provided overrides.

        Args:
            kwargs: Dictionary of configuration overrides

        Returns:
            Updated configuration object
        """
        if not kwargs:
            return self.config

        # Create new config with original values
        updated_config = ComparisonConfig(
            chunk_size=self.config.chunk_size,
            exact_match_only=self.config.exact_match_only,
            ignore_case=self.config.ignore_case,
            ignore_whitespace=self.config.ignore_whitespace,
            float_precision=self.config.float_precision,
            date_match_precision=self.config.date_match_precision,
            null_equality=self.config.null_equality,
        )

        # Update with provided values
        for key, value in kwargs.items():
            if hasattr(updated_config, key):
                setattr(updated_config, key, value)

        return updated_config
