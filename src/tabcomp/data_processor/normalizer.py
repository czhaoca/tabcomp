"""
Enhanced data normalization module for TabComp.

This module provides optimized functionality for standardizing and normalizing data
across different formats and types. It includes improved performance through batch
processing, comprehensive error handling, and strict type safety.

Example:
    >>> normalizer = DataNormalizer()
    >>> normalized_df = normalizer.normalize(df, date_columns=['created_at'])
"""

from typing import Dict, List, Optional, Any, Set, Union, TypeVar, Callable
from typing_extensions import TypedDict
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import re
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

# Type definitions
T = TypeVar("T")
DataFrameish = Union[pd.DataFrame, pd.Series]


class NormalizationError(Exception):
    """Base exception for normalization errors with detailed context."""

    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.column = column
        self.error_type = error_type
        self.details = details or {}


class ValidationResult(TypedDict):
    """Type-safe validation result structure."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class NormalizationConfig:
    """Enhanced configuration for data normalization operations."""

    date_format: str = "%Y-%m-%d"
    numeric_precision: int = 8
    string_case: str = "lower"
    trim_whitespace: bool = True
    replace_nulls: bool = True
    null_replacement: Any = None
    strip_special_chars: bool = True
    batch_size: int = 10000
    max_workers: int = 4
    validation_threshold: float = 0.8
    cache_size: int = 1000
    # Performance tuning
    use_parallel: bool = True
    optimize_dtypes: bool = True
    # Validation settings
    validate_schema: bool = True
    strict_mode: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if not 0 <= self.validation_threshold <= 1:
            raise ValueError("validation_threshold must be between 0 and 1")


class DataNormalizer:
    """
    Enhanced data normalizer with improved performance and type safety.

    Features:
    - Batch processing for large datasets
    - Parallel processing capabilities
    - Comprehensive error handling
    - Memory optimization
    - Type validation and safety
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or NormalizationConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_caches()

    def normalize(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Normalize DataFrame with improved performance and error handling.

        Args:
            df: Input DataFrame to normalize
            date_columns: Optional list of date columns
            numeric_columns: Optional list of numeric columns
            text_columns: Optional list of text columns
            **kwargs: Additional configuration overrides

        Returns:
            Normalized DataFrame

        Raises:
            NormalizationError: If normalization fails
            ValueError: If input validation fails
        """
        try:
            # Validate input
            self._validate_input(df)

            # Update config with overrides
            config = self._update_config(kwargs)

            # Optimize memory usage if configured
            if config.optimize_dtypes:
                df = self._optimize_dtypes(df)

            # Auto-detect column types if not specified
            detected_types = self._detect_column_types(df)
            date_columns = date_columns or detected_types["date"]
            numeric_columns = numeric_columns or detected_types["numeric"]
            text_columns = text_columns or detected_types["text"]

            # Process in batches
            if len(df) > config.batch_size:
                return self._process_in_batches(
                    df, date_columns, numeric_columns, text_columns
                )

            # Process normally for smaller datasets
            return self._normalize_all_columns(
                df, date_columns, numeric_columns, text_columns
            )

        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}")
            raise NormalizationError(
                f"Failed to normalize DataFrame: {str(e)}",
                error_type="global",
                details={"original_error": str(e)},
            )

    def _process_in_batches(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        numeric_columns: List[str],
        text_columns: List[str],
    ) -> pd.DataFrame:
        """Process large DataFrames in batches."""
        chunks = [
            df[i : i + self.config.batch_size]
            for i in range(0, len(df), self.config.batch_size)
        ]

        if self.config.use_parallel and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                normalized_chunks = list(
                    executor.map(
                        lambda chunk: self._normalize_all_columns(
                            chunk, date_columns, numeric_columns, text_columns
                        ),
                        chunks,
                    )
                )
        else:
            normalized_chunks = [
                self._normalize_all_columns(
                    chunk, date_columns, numeric_columns, text_columns
                )
                for chunk in chunks
            ]

        return pd.concat(normalized_chunks, axis=0, ignore_index=True)

    def _normalize_all_columns(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        numeric_columns: List[str],
        text_columns: List[str],
    ) -> pd.DataFrame:
        """Normalize all column types with error handling."""
        result_df = df.copy()

        # Process each column type with proper error handling
        if date_columns:
            result_df = self._normalize_date_columns(result_df, date_columns)
        if numeric_columns:
            result_df = self._normalize_numeric_columns(result_df, numeric_columns)
        if text_columns:
            result_df = self._normalize_text_columns(result_df, text_columns)

        return result_df

    @lru_cache(maxsize=1000)
    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse date string with caching for performance."""
        if pd.isna(value):
            return None

        try:
            return pd.to_datetime(value)
        except Exception:
            return None

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        result_df = df.copy()

        for column in result_df.columns:
            col_type = result_df[column].dtype

            if col_type == "object":
                # Try to convert to categorical for string columns
                nunique = result_df[column].nunique()
                if nunique / len(result_df) < 0.5:  # Less than 50% unique values
                    result_df[column] = result_df[column].astype("category")

            elif col_type == "float64":
                # Downcast float64 to float32 if possible
                result_df[column] = pd.to_numeric(result_df[column], downcast="float")

            elif col_type == "int64":
                # Downcast int64 to smallest possible integer type
                result_df[column] = pd.to_numeric(result_df[column], downcast="integer")

        return result_df

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect column types with improved accuracy."""
        detected = {"date": [], "numeric": [], "text": []}

        for column in df.columns:
            # Skip if column is already numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                detected["numeric"].append(column)
                continue

            # Sample non-null values
            sample = df[column].dropna().head(100)
            if len(sample) == 0:
                continue

            # Try date detection first
            try:
                pd.to_datetime(sample, errors="raise")
                detected["date"].append(column)
                continue
            except (ValueError, TypeError):
                pass

            # Try numeric detection
            try:
                pd.to_numeric(sample, errors="raise")
                detected["numeric"].append(column)
                continue
            except (ValueError, TypeError):
                pass

            # Default to text
            detected["text"].append(column)

        return detected

    def _setup_caches(self) -> None:
        """Initialize caching mechanisms."""
        # Clear any existing cache
        self._parse_date.cache_clear()

        # Initialize other caches as needed
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if len(df) == 0:
            raise ValueError("Input DataFrame is empty")

        if self.config.validate_schema:
            self._validate_schema(df)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema."""
        # Check for duplicate columns
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            raise ValueError(f"Duplicate columns found: {duplicates}")

        # Validate column names
        invalid_chars = r"[^a-zA-Z0-9_]"
        invalid_columns = [col for col in df.columns if re.search(invalid_chars, col)]
        if invalid_columns and self.config.strict_mode:
            raise ValueError(f"Invalid characters in column names: {invalid_columns}")

    def _clear_caches(self) -> None:
        """Clear all caches."""
        self._parse_date.cache_clear()
        self._pattern_cache.clear()
        self._validation_cache.clear()

    def __del__(self):
        """Cleanup resources."""
        self._clear_caches()
