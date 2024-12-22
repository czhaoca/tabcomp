"""
Comparison strategy module for TabComp.

This module implements different strategies for comparing tables, optimizing
for various scenarios such as full table comparison and chunked processing
for large datasets.

Example:
    >>> strategy = ChunkedStrategy(difference_handler, chunk_size=10000)
    >>> differences = strategy.compare(df1, df2, field_mapping, primary_key)
"""

from typing import Dict, List, Set, Any, Protocol, Optional
from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass
import logging

from .difference_handlers import DifferenceHandler
from .exceptions import ComparisonError


@dataclass
class ChunkInfo:
    """Information about a data chunk for processing."""

    start_idx: int
    end_idx: int
    key_range: tuple  # (min_key, max_key)
    size: int


class ComparisonStrategy(ABC):
    """
    Abstract base class for table comparison strategies.

    Defines the interface that all comparison strategies must implement,
    allowing for different approaches to table comparison.
    """

    def __init__(self, difference_handler: DifferenceHandler):
        """
        Initialize the comparison strategy.

        Args:
            difference_handler: Handler for processing differences
        """
        self.difference_handler = difference_handler
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def compare(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compare two tables using the strategy's approach.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_keys: Fields to use as primary keys

        Returns:
            List of difference records

        Raises:
            ComparisonError: If comparison fails
        """
        pass


class FullTableStrategy(ComparisonStrategy):
    """
    Strategy for comparing complete tables in memory.

    Best suited for smaller tables that can fit in memory comfortably.
    Provides fastest comparison for small to medium sized datasets.
    """

    def compare(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compare two tables completely in memory.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_keys: Fields to use as primary keys

        Returns:
            List of difference records

        Raises:
            ComparisonError: If comparison fails
        """
        try:
            # Create sets of primary key values
            keys1 = set(self._get_composite_keys(table1, primary_keys))
            mapped_keys = [field_mapping.get(pk, pk) for pk in primary_keys]
            keys2 = set(self._get_composite_keys(table2, mapped_keys))

            # Find key differences
            missing_keys = keys1 - keys2
            added_keys = keys2 - keys1
            common_keys = keys1 & keys2

            # Process each type of difference
            differences = []
            differences.extend(
                self.difference_handler.process_missing(
                    table1, missing_keys, primary_keys
                )
            )
            differences.extend(
                self.difference_handler.process_added(table2, added_keys, mapped_keys)
            )
            differences.extend(
                self.difference_handler.process_changed(
                    table1, table2, common_keys, field_mapping, primary_keys
                )
            )

            return differences

        except Exception as e:
            raise ComparisonError(f"Full table comparison failed: {str(e)}")

    def _get_composite_keys(
        self, df: pd.DataFrame, key_columns: List[str]
    ) -> List[tuple]:
        """
        Create composite keys from multiple columns.

        Args:
            df: DataFrame to process
            key_columns: Columns to combine for key

        Returns:
            List of composite key tuples
        """
        return [tuple(row) for row in df[key_columns].itertuples(index=False)]


class ChunkedStrategy(ComparisonStrategy):
    """
    Strategy for comparing tables in chunks.

    Suitable for large tables that shouldn't be processed entirely in memory.
    Trades some performance for lower memory usage.
    """

    def __init__(self, difference_handler: DifferenceHandler, chunk_size: int = 10000):
        """
        Initialize the chunked comparison strategy.

        Args:
            difference_handler: Handler for processing differences
            chunk_size: Number of rows per chunk
        """
        super().__init__(difference_handler)
        self.chunk_size = chunk_size

    def compare(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compare two tables in chunks.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_keys: Fields to use as primary keys

        Returns:
            List of difference records

        Raises:
            ComparisonError: If comparison fails
        """
        try:
            differences = []

            # Sort tables by primary keys for efficient chunking
            table1_sorted = table1.sort_values(by=primary_keys)
            mapped_keys = [field_mapping.get(pk, pk) for pk in primary_keys]
            table2_sorted = table2.sort_values(by=mapped_keys)

            # Process in chunks
            for chunk_info in self._generate_chunks(table1_sorted, primary_keys):
                # Get the chunk from table1
                chunk1 = table1_sorted.iloc[chunk_info.start_idx : chunk_info.end_idx]

                # Find corresponding chunk in table2
                chunk2 = self._get_matching_chunk(
                    table2_sorted, chunk1, field_mapping, primary_keys
                )

                # Compare chunks using full table strategy
                chunk_differences = FullTableStrategy(self.difference_handler).compare(
                    chunk1, chunk2, field_mapping, primary_keys
                )
                differences.extend(chunk_differences)

            return differences

        except Exception as e:
            raise ComparisonError(f"Chunked comparison failed: {str(e)}")

    def _generate_chunks(
        self, df: pd.DataFrame, key_columns: List[str]
    ) -> List[ChunkInfo]:
        """
        Generate chunk information for processing.

        Args:
            df: DataFrame to chunk
            key_columns: Primary key columns

        Returns:
            List of ChunkInfo objects
        """
        chunks = []
        total_rows = len(df)

        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]

            # Get key range for chunk
            min_keys = chunk_df[key_columns].min()
            max_keys = chunk_df[key_columns].max()
            key_range = (
                tuple(min_keys[col] for col in key_columns),
                tuple(max_keys[col] for col in key_columns),
            )

            chunks.append(
                ChunkInfo(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_range=key_range,
                    size=end_idx - start_idx,
                )
            )

        return chunks

    def _get_matching_chunk(
        self,
        table2: pd.DataFrame,
        chunk1: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> pd.DataFrame:
        """
        Get the relevant chunk from table2 that corresponds to chunk1's key range.

        Args:
            table2: Second table
            chunk1: Chunk from first table
            field_mapping: Field mapping dictionary
            primary_keys: Primary key fields

        Returns:
            DataFrame containing relevant portion of table2
        """
        if len(chunk1) == 0:
            return pd.DataFrame()

        # Get min and max keys from chunk1
        min_keys = chunk1[primary_keys].min()
        max_keys = chunk1[primary_keys].max()

        # Map keys to table2 column names
        table2_keys = [field_mapping.get(pk, pk) for pk in primary_keys]

        # Create mask for table2
        mask = pd.Series(True, index=table2.index)
        for pk1, pk2 in zip(primary_keys, table2_keys):
            mask &= (table2[pk2] >= min_keys[pk1]) & (table2[pk2] <= max_keys[pk1])

        return table2[mask]


class HybridStrategy(ComparisonStrategy):
    """
    Strategy combining full table and chunked approaches.

    Uses heuristics to choose the best strategy based on table characteristics.
    Attempts to optimize both memory usage and performance.
    """

    def __init__(
        self,
        difference_handler: DifferenceHandler,
        memory_threshold: int = 1_000_000,  # 1M rows
        chunk_size: int = 10000,
    ):
        """
        Initialize the hybrid comparison strategy.

        Args:
            difference_handler: Handler for processing differences
            memory_threshold: Row count threshold for chunking
            chunk_size: Number of rows per chunk if chunking
        """
        super().__init__(difference_handler)
        self.memory_threshold = memory_threshold
        self.chunk_size = chunk_size
        self.full_strategy = FullTableStrategy(difference_handler)
        self.chunked_strategy = ChunkedStrategy(difference_handler, chunk_size)

    def compare(
        self,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        field_mapping: Dict[str, str],
        primary_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compare tables using the most appropriate strategy.

        Args:
            table1: First DataFrame to compare
            table2: Second DataFrame to compare
            field_mapping: Dictionary mapping fields from table1 to table2
            primary_keys: Fields to use as primary keys

        Returns:
            List of difference records

        Raises:
            ComparisonError: If comparison fails
        """
        try:
            # Choose strategy based on table sizes
            total_rows = max(len(table1), len(table2))

            if total_rows <= self.memory_threshold:
                self.logger.info("Using full table strategy")
                return self.full_strategy.compare(
                    table1, table2, field_mapping, primary_keys
                )
            else:
                self.logger.info("Using chunked strategy")
                return self.chunked_strategy.compare(
                    table1, table2, field_mapping, primary_keys
                )

        except Exception as e:
            raise ComparisonError(f"Hybrid comparison failed: {str(e)}")
