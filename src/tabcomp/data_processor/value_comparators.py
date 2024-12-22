"""
Value comparison module for TabComp.

This module provides specialized comparison logic for different data types,
handling the nuances of comparing numeric values, dates, strings, and NULL values
according to configurable rules.

Example:
    >>> comparator = ValueComparator(config)
    >>> is_equal = comparator.values_equal(value1, value2)
"""

from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging

from .exceptions import ComparisonError


@dataclass
class ValueComparisonConfig:
    """Configuration for value comparison operations."""

    numeric_precision: int = 8
    date_precision: str = "second"  # "year", "month", "day", "hour", "minute", "second"
    string_case_sensitive: bool = False
    string_ignore_whitespace: bool = True
    null_equality: bool = True  # True: NULL == NULL, False: NULL != NULL
    inf_equality: bool = True  # True: inf == inf, False: inf != inf
    exact_match_only: bool = False
    ignore_string_type_mismatch: bool = True  # int vs str comparison
    float_tolerance: float = 1e-8
    datetime_tolerance: Optional[timedelta] = None


class ValueComparator:
    """
    Handles comparison of individual values across different data types.

    This class provides methods for comparing values while handling type
    mismatches, NULL values, and precision requirements.
    """

    def __init__(self, config: Optional[ValueComparisonConfig] = None):
        """
        Initialize the ValueComparator.

        Args:
            config: Configuration options for value comparison.
                   If not provided, uses default values.
        """
        self.config = config or ValueComparisonConfig()
        self.logger = logging.getLogger(__name__)

    def values_equal(
        self, value1: Any, value2: Any, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Compare two values according to configured rules.

        Args:
            value1: First value to compare
            value2: Second value to compare
            context: Optional comparison context (e.g., field name, types)

        Returns:
            True if values are considered equal, False otherwise
        """
        try:
            # Handle NULL values
            if pd.isna(value1) or pd.isna(value2):
                return self._compare_null_values(value1, value2)

            # Handle exact match requirement
            if self.config.exact_match_only:
                return value1 == value2

            # Handle type-specific comparisons
            if isinstance(value1, (int, float)) or isinstance(value2, (int, float)):
                return self._compare_numeric_values(value1, value2)
            elif isinstance(value1, (datetime, pd.Timestamp)) or isinstance(
                value2, (datetime, pd.Timestamp)
            ):
                return self._compare_datetime_values(value1, value2)
            elif isinstance(value1, str) or isinstance(value2, str):
                return self._compare_string_values(value1, value2)

            # Default comparison
            return value1 == value2

        except Exception as e:
            self.logger.warning(
                f"Error comparing values: {str(e)}",
                extra={"value1": value1, "value2": value2, "context": context},
            )
            return False

    def _compare_null_values(self, value1: Any, value2: Any) -> bool:
        """
        Compare NULL/None/NaN values.

        Args:
            value1: First value
            value2: Second value

        Returns:
            True if values are considered equal according to NULL rules
        """
        if pd.isna(value1) and pd.isna(value2):
            return self.config.null_equality
        return False

    def _compare_numeric_values(self, value1: Any, value2: Any) -> bool:
        """
        Compare numeric values with configurable precision.

        Args:
            value1: First value
            value2: Second value

        Returns:
            True if values are considered numerically equal
        """
        try:
            # Handle infinity comparisons
            if np.isinf(value1) or np.isinf(value2):
                if not self.config.inf_equality:
                    return False
                return value1 == value2

            # Convert to float for comparison
            float1 = float(value1)
            float2 = float(value2)

            # Use numpy's isclose for float comparison
            return np.isclose(
                float1,
                float2,
                rtol=self.config.float_tolerance,
                atol=0,
                equal_nan=False,
            )

        except (ValueError, TypeError):
            if self.config.ignore_string_type_mismatch:
                # Try comparing string representations
                return self._compare_string_values(str(value1), str(value2))
            return False

    def _compare_datetime_values(self, value1: Any, value2: Any) -> bool:
        """
        Compare datetime values with configurable precision.

        Args:
            value1: First datetime value
            value2: Second datetime value

        Returns:
            True if datetimes are considered equal
        """
        try:
            # Convert to pandas Timestamp for consistent handling
            ts1 = pd.Timestamp(value1)
            ts2 = pd.Timestamp(value2)

            # Apply configured precision
            if self.config.date_precision == "year":
                return ts1.year == ts2.year
            elif self.config.date_precision == "month":
                return (ts1.year, ts1.month) == (ts2.year, ts2.month)
            elif self.config.date_precision == "day":
                return ts1.date() == ts2.date()
            elif self.config.date_precision == "hour":
                return ts1.floor("H") == ts2.floor("H")
            elif self.config.date_precision == "minute":
                return ts1.floor("T") == ts2.floor("T")
            else:  # second
                return ts1.floor("S") == ts2.floor("S")

        except (ValueError, TypeError):
            if self.config.ignore_string_type_mismatch:
                # Try comparing string representations
                return self._compare_string_values(str(value1), str(value2))
            return False

    def _compare_string_values(self, value1: Any, value2: Any) -> bool:
        """
        Compare string values with configurable case and whitespace handling.

        Args:
            value1: First string value
            value2: Second string value

        Returns:
            True if strings are considered equal
        """
        try:
            # Convert to strings
            str1 = str(value1)
            str2 = str(value2)

            # Apply whitespace normalization
            if self.config.string_ignore_whitespace:
                str1 = str1.strip()
                str2 = str2.strip()

            # Apply case normalization
            if not self.config.string_case_sensitive:
                str1 = str1.lower()
                str2 = str2.lower()

            return str1 == str2

        except (ValueError, TypeError):
            return False

    def get_comparison_info(self, value1: Any, value2: Any) -> Dict[str, Any]:
        """
        Get detailed information about value comparison.

        Args:
            value1: First value
            value2: Second value

        Returns:
            Dictionary containing comparison details
        """
        info = {
            "equal": self.values_equal(value1, value2),
            "type1": str(type(value1).__name__),
            "type2": str(type(value2).__name__),
            "is_null1": pd.isna(value1),
            "is_null2": pd.isna(value2),
        }

        # Add type-specific details
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            try:
                info.update(
                    {
                        "absolute_difference": abs(float(value1) - float(value2)),
                        "percentage_difference": self._calculate_percentage_difference(
                            float(value1), float(value2)
                        ),
                    }
                )
            except (ValueError, TypeError):
                pass

        elif isinstance(value1, (datetime, pd.Timestamp)) and isinstance(
            value2, (datetime, pd.Timestamp)
        ):
            try:
                diff = pd.Timestamp(value1) - pd.Timestamp(value2)
                info["time_difference_seconds"] = diff.total_seconds()
            except (ValueError, TypeError):
                pass

        elif isinstance(value1, str) and isinstance(value2, str):
            info.update(
                {
                    "length_difference": len(str(value1)) - len(str(value2)),
                    "case_difference": value1.lower() == value2.lower()
                    and value1 != value2,
                }
            )

        return info

    def _calculate_percentage_difference(self, value1: float, value2: float) -> float:
        """
        Calculate percentage difference between two numeric values.

        Args:
            value1: First value
            value2: Second value

        Returns:
            Percentage difference or inf for certain cases
        """
        if value1 == value2:
            return 0.0
        if value1 == 0 and value2 == 0:
            return 0.0
        if value1 == 0:
            return float("inf")
        return abs((value2 - value1) / value1) * 100
