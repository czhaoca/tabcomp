"""
Difference handling module for TabComp.

This module provides functionality for processing and categorizing different
types of differences found between tables during comparison operations.
It handles missing records, added records, and value changes.

Example:
    >>> handler = DifferenceHandler(value_comparator)
    >>> differences = handler.process_changed(table1, table2, common_keys)
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

from .value_comparators import ValueComparator
from .exceptions import ComparisonError


@dataclass
class DifferenceRecord:
    """Record of a single difference between tables."""

    key_values: Dict[str, Any]
    field_name: str
    table1_value: Any
    table2_value: Any
    difference_type: str  # "missing", "added", "changed"
    difference_details: Optional[Dict[str, Any]] = None


class DifferenceHandler:
    """
    Handles processing and categorization of table differences.

    This class provides methods for handling different types of
    differences found during table comparison operations.
    """

    def __init__(self, value_comparator: ValueComparator):
        """
        Initialize the DifferenceHandler.

        Args:
            value_comparator: Component for comparing individual values
        """
        self.value_comparator = value_comparator
        self._cached_key_rows: Dict[Tuple[str, tuple], Dict[str, Any]] = {}

    def process_missing(
        self, table1: pd.DataFrame, missing_keys: Set[tuple], primary_keys: List[str]
    ) -> List[DifferenceRecord]:
        """
        Process records that exist in table1 but not in table2.

        Args:
            table1: First DataFrame
            missing_keys: Set of primary key values missing from table2
            primary_keys: Primary key field names

        Returns:
            List of DifferenceRecord objects for missing records
        """
        differences = []
        for key_values in missing_keys:
            key_dict = dict(zip(primary_keys, key_values))

            # Get the row for this key
            row = self._get_row_by_key(table1, primary_keys, key_values)
            if row is None:
                continue

            # Create a difference record for each non-key field
            for col in table1.columns:
                if col not in primary_keys:
                    differences.append(
                        DifferenceRecord(
                            key_values=key_dict.copy(),
                            field_name=col,
                            table1_value=row[col],
                            table2_value=None,
                            difference_type="missing",
                            difference_details={
                                "message": "Record exists in table1 but not in table2"
                            },
                        )
                    )

        return differences

    def process_added(
        self, table2: pd.DataFrame, added_keys: Set[tuple], primary_keys: List[str]
    ) -> List[DifferenceRecord]:
        """
        Process records that exist in table2 but not in table1.

        Args:
            table2: Second DataFrame
            added_keys: Set of primary key values not in table1
            primary_keys: Primary key field names

        Returns:
            List of DifferenceRecord objects for added records
        """
        differences = []
        for key_values in added_keys:
            key_dict = dict(zip(primary_keys, key_values))

            # Get the row for this key
            row = self._get_row_by_key(table2, primary_keys, key_values)
            if row is None:
                continue

            # Create a difference record for each non-key field
            for col in table2.columns:
                if col not in primary_keys:
                    differences.append(
                        DifferenceRecord(
                            key_values=key_dict.copy(),
                            field_name=col,
                            table1_value=None,
                            table2_value=row[col],
                            difference_type="added",
                            difference_details={
                                "message": "Record exists in table2 but not in table1"
                            },
                        )
                    )

        return differences

    def process_changed(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        common_keys: Set[tuple],
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> List[DifferenceRecord]:
        """
        Process records that exist in both tables but have differences.

        Args:
            table1: First DataFrame
            table2: Second DataFrame
            common_keys: Set of primary key values present in both tables
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_keys: Primary key field names

        Returns:
            List of DifferenceRecord objects for changed records
        """
        differences = []
        mapped_keys = [field_mapping.get(pk, pk) for pk in primary_keys]

        for key_values in common_keys:
            key_dict = dict(zip(primary_keys, key_values))

            # Get the rows for this key from both tables
            row1 = self._get_row_by_key(table1, primary_keys, key_values)
            row2 = self._get_row_by_key(table2, mapped_keys, key_values)

            if row1 is None or row2 is None:
                continue

            # Compare non-key fields
            for field1, field2 in field_mapping.items():
                if field1 not in primary_keys:
                    value1 = row1[field1]
                    value2 = row2[field2]

                    if not self.value_comparator.values_equal(value1, value2):
                        diff_details = self._get_difference_details(
                            value1, value2, field1
                        )
                        differences.append(
                            DifferenceRecord(
                                key_values=key_dict.copy(),
                                field_name=field1,
                                table1_value=value1,
                                table2_value=value2,
                                difference_type="changed",
                                difference_details=diff_details,
                            )
                        )

        return differences

    def _get_row_by_key(
        self, df: pd.DataFrame, key_fields: List[str], key_values: tuple
    ) -> Optional[pd.Series]:
        """
        Get a row from DataFrame by its primary key values.

        Args:
            df: DataFrame to search
            key_fields: Primary key field names
            key_values: Values of primary keys to match

        Returns:
            Matching row as Series or None if not found
        """
        # Check cache first
        cache_key = (df.shape[0], key_values)
        if cache_key in self._cached_key_rows:
            return self._cached_key_rows[cache_key]

        # Create mask for key matching
        mask = pd.Series(True, index=df.index)
        for field, value in zip(key_fields, key_values):
            mask &= df[field] == value

        # Get matching row
        matching_rows = df[mask]
        if len(matching_rows) == 0:
            return None
        if len(matching_rows) > 1:
            raise ComparisonError(f"Multiple rows found for key values: {key_values}")

        # Cache and return result
        row = matching_rows.iloc[0]
        self._cached_key_rows[cache_key] = row
        return row

    def _get_difference_details(
        self, value1: Any, value2: Any, field_name: str
    ) -> Dict[str, Any]:
        """
        Generate detailed information about a value difference.

        Args:
            value1: Value from first table
            value2: Value from second table
            field_name: Name of the field being compared

        Returns:
            Dictionary containing difference details
        """
        details = {
            "message": "Values differ between tables",
            "field_name": field_name,
            "type1": str(type(value1).__name__),
            "type2": str(type(value2).__name__),
        }

        # Add type-specific details
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            details.update(
                {
                    "absolute_difference": abs(float(value1) - float(value2)),
                    "percentage_difference": self._calculate_percentage_difference(
                        float(value1), float(value2)
                    ),
                }
            )
        elif isinstance(value1, str) and isinstance(value2, str):
            details.update(
                {
                    "levenshtein_distance": self._calculate_levenshtein_distance(
                        str(value1), str(value2)
                    )
                }
            )
        elif pd.isna(value1) != pd.isna(value2):
            details["message"] = "One value is NULL while the other is not"

        return details

    def _calculate_percentage_difference(self, value1: float, value2: float) -> float:
        """
        Calculate percentage difference between two numeric values.

        Args:
            value1: First value
            value2: Second value

        Returns:
            Percentage difference
        """
        if value1 == value2:
            return 0.0
        if value1 == 0 and value2 == 0:
            return 0.0
        if value1 == 0:
            return float("inf")
        return abs((value2 - value1) / value1) * 100

    def _calculate_levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Edit distance between strings
        """
        if str1 == str2:
            return 0

        # Create matrix of zeros
        rows = len(str1) + 1
        cols = len(str2) + 1
        matrix = np.zeros((rows, cols), dtype=int)

        # Initialize first row and column
        for i in range(rows):
            matrix[i, 0] = i
        for j in range(cols):
            matrix[0, j] = j

        # Fill rest of the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if str1[i - 1] == str2[j - 1]:
                    matrix[i, j] = matrix[i - 1, j - 1]
                else:
                    matrix[i, j] = min(
                        matrix[i - 1, j] + 1,  # deletion
                        matrix[i, j - 1] + 1,  # insertion
                        matrix[i - 1, j - 1] + 1,  # substitution
                    )

        return matrix[rows - 1, cols - 1]

    def _clear_cache(self) -> None:
        """Clear the row cache to free memory."""
        self._cached_key_rows.clear()
