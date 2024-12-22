"""
Configuration module for TabComp Data Processor.

This module contains configuration classes and validation logic for
the data processor components. It provides type-safe configuration
with validation and sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from enum import Enum


class DatePrecision(Enum):
    """Supported date comparison precision levels."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"


class StringCase(Enum):
    """String case handling options."""

    LOWER = "lower"
    UPPER = "upper"
    PRESERVE = "preserve"


@dataclass
class ValidationConfig:
    """Configuration for data validation operations."""

    require_unique_keys: bool = True
    allow_missing_columns: bool = False
    strict_schema_check: bool = True
    schema_match_threshold: float = 0.8


@dataclass
class ComparisonOptions:
    """Configuration options for table comparison."""

    # General options
    chunk_size: int = field(default=10000)
    memory_threshold_mb: int = field(default=500)
    parallel_threshold: int = field(default=100000)

    # Comparison behavior
    case_sensitive: bool = field(default=False)
    ignore_whitespace: bool = field(default=True)
    exact_match_only: bool = field(default=False)
    custom_match_threshold: float = field(default=0.85)

    # Type-specific options
    date_format: str = field(default="%Y-%m-%d")
    date_precision: DatePrecision = field(default=DatePrecision.SECOND)
    numeric_precision: int = field(default=8)
    string_case: StringCase = field(default=StringCase.LOWER)

    # Column handling
    ignore_columns: List[str] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    column_aliases: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate_numeric_values()
        self._validate_thresholds()
        self._validate_formats()

    def _validate_numeric_values(self):
        """Validate numeric configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.memory_threshold_mb <= 0:
            raise ValueError("memory_threshold_mb must be positive")
        if self.parallel_threshold <= 0:
            raise ValueError("parallel_threshold must be positive")
        if self.numeric_precision < 0:
            raise ValueError("numeric_precision cannot be negative")

    def _validate_thresholds(self):
        """Validate threshold values."""
        if not 0 <= self.custom_match_threshold <= 1:
            raise ValueError("custom_match_threshold must be between 0 and 1")

    def _validate_formats(self):
        """Validate format strings and enums."""
        try:
            # Validate date format
            datetime.now().strftime(self.date_format)
        except ValueError as e:
            raise ValueError(f"Invalid date_format: {str(e)}")


@dataclass
class ComparisonResult:
    """Results of table comparison operation."""

    # Basic metrics
    total_rows: int
    total_differences: int
    execution_time: float
    memory_usage: int

    # Detailed results
    differences: pd.DataFrame = field(repr=False)
    summary: Dict[str, Any] = field(repr=False)

    # Metadata
    comparison_timestamp: datetime = field(default_factory=datetime.now)
    options_used: ComparisonOptions = field(repr=False)

    @property
    def difference_rate(self) -> float:
        """Calculate the rate of differences found."""
        return self.total_differences / self.total_rows if self.total_rows > 0 else 0

    @property
    def processing_rate(self) -> float:
        """Calculate rows processed per second."""
        return self.total_rows / self.execution_time if self.execution_time > 0 else 0

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the comparison."""
        return {
            "total_rows": self.total_rows,
            "total_differences": self.total_differences,
            "difference_rate": self.difference_rate,
            "processing_rate": self.processing_rate,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "timestamp": self.comparison_timestamp,
        }


@dataclass
class MatchResult:
    """Results of field matching operation."""

    matched_fields: Dict[str, str]
    unmatched_fields1: List[str]
    unmatched_fields2: List[str]
    match_confidence: Dict[str, float]
    suggested_matches: Dict[str, List[tuple[str, float]]]

    @property
    def match_rate(self) -> float:
        """Calculate the rate of successfully matched fields."""
        total_fields = len(self.matched_fields) + len(self.unmatched_fields1)
        return len(self.matched_fields) / total_fields if total_fields > 0 else 0

    def get_match_summary(self) -> Dict[str, Any]:
        """Get summary of the matching operation."""
        return {
            "total_matched": len(self.matched_fields),
            "total_unmatched1": len(self.unmatched_fields1),
            "total_unmatched2": len(self.unmatched_fields2),
            "match_rate": self.match_rate,
            "average_confidence": (
                sum(self.match_confidence.values()) / len(self.match_confidence)
                if self.match_confidence
                else 0
            ),
        }
