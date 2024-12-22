"""
File parsing module for TabComp.

This module handles parsing of CSV and Excel files, with support for
large file processing and multi-sheet workbooks.
"""

import os
from typing import Optional, Dict, List, Iterator, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from .exceptions import (
    FileNotFoundError,
    InvalidFormatError,
    InvalidStructureError,
    DecodeError,
)
from .encoding import EncodingDetector


@dataclass
class ParsedData:
    """Contains parsed file data and metadata."""

    data: pd.DataFrame
    sheet_name: Optional[str]
    total_rows: int
    total_columns: int
    column_types: Dict[str, str]
    null_counts: Dict[str, int]


@dataclass
class ParseOptions:
    """Configuration options for file parsing."""

    encoding: Optional[str] = None
    sheet_name: Optional[str] = None
    chunk_size: int = 10000
    na_values: List[str] = None
    parse_dates: List[str] = None
    dtype_backend: str = "numpy"  # or 'pyarrow'
    date_format: Optional[str] = None


class FileParser:
    """Handles parsing of CSV and Excel files."""

    def __init__(self, encoding_detector: Optional[EncodingDetector] = None):
        """
        Initialize the FileParser.

        Args:
            encoding_detector: Optional EncodingDetector instance.
                             Creates new instance if not provided.
        """
        self.encoding_detector = encoding_detector or EncodingDetector()

    def parse_file(
        self, file_path: str, options: Optional[ParseOptions] = None
    ) -> ParsedData:
        """
        Parses a file and returns structured data.

        Args:
            file_path: Path to the file to parse
            options: Optional parsing configuration options

        Returns:
            ParsedData containing the parsed data and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            InvalidFormatError: If the file format is not supported
            DecodeError: If there are encoding issues
            InvalidStructureError: If the file structure is invalid
        """
        options = options or ParseOptions()

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        # Determine file type and parse accordingly
        file_extension = Path(file_path).suffix.lower()
        if file_extension == ".csv":
            return self._parse_csv(file_path, options)
        elif file_extension in {".xlsx", ".xls"}:
            return self._parse_excel(file_path, options)
        else:
            raise InvalidFormatError(file_path, file_extension)

    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Gets list of sheet names from an Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of sheet names

        Raises:
            InvalidFormatError: If the file is not an Excel file
        """
        if not file_path.lower().endswith((".xlsx", ".xls")):
            raise InvalidFormatError(
                file_path,
                Path(file_path).suffix,
                details={"error": "Not an Excel file"},
            )

        try:
            return pd.ExcelFile(file_path).sheet_names
        except Exception as e:
            raise InvalidStructureError(
                file_path, details={"error": f"Failed to read Excel sheets: {str(e)}"}
            )

    def read_chunks(
        self,
        file_path: str,
        chunk_size: int = 10000,
        options: Optional[ParseOptions] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Reads file in chunks to handle large files.

        Args:
            file_path: Path to the file to read
            chunk_size: Number of rows per chunk
            options: Optional parsing configuration

        Yields:
            DataFrame chunks of the specified size

        Raises:
            FileNotFoundError: If the file doesn't exist
            InvalidFormatError: If the file format is not supported
        """
        options = options or ParseOptions(chunk_size=chunk_size)

        if file_path.lower().endswith(".csv"):
            yield from self._read_csv_chunks(file_path, options)
        else:
            # Excel files are read in memory, so we chunk the loaded data
            data = self._parse_excel(file_path, options).data
            for i in range(0, len(data), chunk_size):
                yield data.iloc[i : i + chunk_size]

    def _parse_csv(self, file_path: str, options: ParseOptions) -> ParsedData:
        """
        Parses a CSV file.

        Args:
            file_path: Path to the CSV file
            options: Parsing configuration options

        Returns:
            ParsedData containing parsed CSV data and metadata

        Raises:
            DecodeError: If there are encoding issues
            InvalidStructureError: If the CSV structure is invalid
        """
        try:
            # Detect encoding if not specified
            encoding = options.encoding
            if not encoding:
                encoding = self.encoding_detector.detect_encoding(file_path).encoding

            # Read the CSV file
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                na_values=options.na_values,
                parse_dates=options.parse_dates,
                date_format=options.date_format,
            )

            return self._create_parsed_data(df)

        except UnicodeError as e:
            raise DecodeError(file_path, encoding, details={"error": str(e)})
        except Exception as e:
            raise InvalidStructureError(file_path, details={"error": str(e)})

    def _parse_excel(self, file_path: str, options: ParseOptions) -> ParsedData:
        """
        Parses an Excel file.

        Args:
            file_path: Path to the Excel file
            options: Parsing configuration options

        Returns:
            ParsedData containing parsed Excel data and metadata

        Raises:
            InvalidStructureError: If the Excel structure is invalid
        """
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=options.sheet_name,
                na_values=options.na_values,
                parse_dates=options.parse_dates,
                date_format=options.date_format,
            )

            # Handle multiple sheets if no specific sheet was requested
            if isinstance(df, dict) and options.sheet_name is None:
                # Default to first sheet
                sheet_name = list(df.keys())[0]
                df = df[sheet_name]
            else:
                sheet_name = options.sheet_name

            return self._create_parsed_data(df, sheet_name)

        except Exception as e:
            raise InvalidStructureError(file_path, details={"error": str(e)})

    def _read_csv_chunks(
        self, file_path: str, options: ParseOptions
    ) -> Iterator[pd.DataFrame]:
        """
        Reads a CSV file in chunks.

        Args:
            file_path: Path to the CSV file
            options: Parsing configuration options

        Yields:
            DataFrame chunks of the specified size
        """
        try:
            # Detect encoding if not specified
            encoding = options.encoding
            if not encoding:
                encoding = self.encoding_detector.detect_encoding(file_path).encoding

            # Create chunk iterator
            chunks = pd.read_csv(
                file_path,
                encoding=encoding,
                chunksize=options.chunk_size,
                na_values=options.na_values,
                parse_dates=options.parse_dates,
                date_format=options.date_format,
            )

            yield from chunks

        except UnicodeError as e:
            raise DecodeError(file_path, encoding, details={"error": str(e)})
        except Exception as e:
            raise InvalidStructureError(file_path, details={"error": str(e)})

    def _create_parsed_data(
        self, df: pd.DataFrame, sheet_name: Optional[str] = None
    ) -> ParsedData:
        """
        Creates ParsedData from a DataFrame.

        Args:
            df: DataFrame to process
            sheet_name: Optional name of the Excel sheet

        Returns:
            ParsedData containing the DataFrame and metadata
        """
        return ParsedData(
            data=df,
            sheet_name=sheet_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            column_types=dict(df.dtypes.astype(str)),
            null_counts=dict(df.isnull().sum()),
        )
