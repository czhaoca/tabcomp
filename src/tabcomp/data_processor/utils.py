"""
Utility functions for the TabComp Data Processor module.

This module provides shared utility functions used across different components
of the data processor, including type checking, data validation, memory
management, and performance monitoring utilities.

Example:
    >>> from tabcomp.data_processor.utils import estimate_memory_usage
    >>> mem_bytes = estimate_memory_usage(df)
"""

from typing import Any, Dict, List, Optional, Set, Union, TypeVar
import pandas as pd
import numpy as np
import psutil
import logging
from datetime import datetime
import re
from contextlib import contextmanager
import time

# Type variable for generic functions
T = TypeVar("T")


class MemoryTracker:
    """
    Tracks memory usage during operations.

    This class provides utilities for monitoring and managing memory usage
    during data processing operations.
    """

    def __init__(self):
        """Initialize the memory tracker."""
        self.initial_memory = self.get_current_memory()
        self.peak_memory = self.initial_memory
        self.start_time = time.time()

    def get_current_memory(self) -> int:
        """
        Get current memory usage of the process.

        Returns:
            Current memory usage in bytes
        """
        return psutil.Process().memory_info().rss

    def update(self) -> None:
        """Update peak memory usage."""
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary containing memory usage statistics
        """
        current = self.get_current_memory()
        return {
            "initial_memory": self.initial_memory,
            "current_memory": current,
            "peak_memory": self.peak_memory,
            "memory_increase": current - self.initial_memory,
            "duration": time.time() - self.start_time,
        }


@contextmanager
def track_memory():
    """
    Context manager for tracking memory usage.

    Example:
        >>> with track_memory() as tracker:
        ...     # Perform memory-intensive operations
        ...     stats = tracker.get_stats()
    """
    tracker = MemoryTracker()
    try:
        yield tracker
    finally:
        tracker.update()


def estimate_memory_usage(df: pd.DataFrame) -> int:
    """
    Estimate memory usage of a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Estimated memory usage in bytes
    """
    return df.memory_usage(deep=True).sum()


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks of specified size.

    Args:
        df: DataFrame to split
        chunk_size: Number of rows per chunk

    Returns:
        List of DataFrame chunks
    """
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


class TypeValidator:
    """
    Provides utilities for type checking and validation.
    """

    @staticmethod
    def is_date_like(value: Any) -> bool:
        """
        Check if a value appears to be a date.

        Args:
            value: Value to check

        Returns:
            True if value appears to be a date
        """
        return isinstance(value, (datetime, pd.Timestamp)) or (
            isinstance(value, str) and is_date_string(value)
        )

    @staticmethod
    def is_numeric_like(value: Any) -> bool:
        """
        Check if a value appears to be numeric.

        Args:
            value: Value to check

        Returns:
            True if value appears to be numeric
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def coerce_numeric(value: Any, default: Optional[float] = None) -> Optional[float]:
        """
        Attempt to convert value to numeric type.

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            Converted numeric value or default
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


def is_date_string(value: str) -> bool:
    """
    Check if a string appears to be a date.

    Args:
        value: String to check

    Returns:
        True if string appears to be a date
    """
    common_date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
        r"^\d{2}/\d{2}/\d{4}$",  # DD/MM/YYYY or MM/DD/YYYY
        r"^\d{4}/\d{2}/\d{2}$",  # YYYY/MM/DD
        r"^\d{2}-\d{2}-\d{4}$",  # DD-MM-YYYY or MM-DD-YYYY
    ]
    return any(re.match(pattern, value) for pattern in common_date_patterns)


def normalize_column_name(name: str) -> str:
    """
    Normalize column name for consistent comparison.

    Args:
        name: Column name to normalize

    Returns:
        Normalized column name
    """
    # Convert to lowercase
    normalized = name.lower()

    # Replace special characters with underscore
    normalized = re.sub(r"[^\w\s]", "_", normalized)

    # Replace multiple underscores with single
    normalized = re.sub(r"_+", "_", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized


def get_overlap_score(str1: str, str2: str) -> float:
    """
    Calculate overlap score between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Overlap score between 0 and 1
    """
    set1 = set(str1.lower().split("_"))
    set2 = set(str2.lower().split("_"))

    if not set1 or not set2:
        return 0.0

    overlap = len(set1 & set2)
    total = len(set1 | set2)

    return overlap / total


def are_similar_numbers(value1: float, value2: float, tolerance: float = 1e-8) -> bool:
    """
    Check if two numbers are similar within tolerance.

    Args:
        value1: First number
        value2: Second number
        tolerance: Relative tolerance for comparison

    Returns:
        True if numbers are considered similar
    """
    return np.isclose(value1, value2, rtol=tolerance)


def safe_cast(value: Any, cast_type: type) -> Optional[Any]:
    """
    Safely cast value to specified type.

    Args:
        value: Value to cast
        cast_type: Type to cast to

    Returns:
        Cast value or None if casting fails
    """
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return None


@contextmanager
def timer(operation_name: str):
    """
    Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed

    Example:
        >>> with timer('data_processing'):
        ...     # Perform operation
        ...     pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logging.info(f"{operation_name} completed in {duration:.2f} seconds")


class ProgressLogger:
    """
    Utility for logging progress of long operations.
    """

    def __init__(self, total: int, log_interval: int = 1000):
        """
        Initialize progress logger.

        Args:
            total: Total number of items to process
            log_interval: How often to log progress (in number of items)
        """
        self.total = total
        self.current = 0
        self.log_interval = log_interval
        self.start_time = time.time()

    def update(self, amount: int = 1) -> None:
        """
        Update progress counter.

        Args:
            amount: Number of items processed
        """
        self.current += amount
        if self.current % self.log_interval == 0:
            self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        percentage = (self.current / self.total) * 100
        rate = self.current / elapsed if elapsed > 0 else 0

        logging.info(
            f"Progress: {percentage:.1f}% ({self.current}/{self.total}) "
            f"Rate: {rate:.1f} items/sec"
        )


def clean_string(s: str) -> str:
    """
    Clean string by removing special characters and normalizing whitespace.

    Args:
        s: String to clean

    Returns:
        Cleaned string
    """
    # Remove special characters
    cleaned = re.sub(r"[^\w\s]", " ", s)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned.strip().lower()
