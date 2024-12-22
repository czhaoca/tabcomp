"""
Unit tests for the FileParser component.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from tabcomp.file_handler.exceptions import (
    FileNotFoundError,
    InvalidFormatError,
    InvalidStructureError,
    DecodeError,
)
from tabcomp.file_handler.parser import FileParser, ParsedData, ParseOptions


class TestFileParser:
    """Test suite for FileParser class."""

    def test_parse_csv_success(self, temp_csv_file: Path) -> None:
        """Test successful CSV parsing."""
        parser = FileParser()
        result = parser.parse_file(str(temp_csv_file))
        assert isinstance(result, ParsedData)
        assert result.total_rows == 3
        assert result.total_columns == 3
        assert "id" in result.column_types
        assert result.sheet_name is None

    def test_parse_excel_success(self, temp_excel_file: Path) -> None:
        """Test successful Excel parsing."""
        parser = FileParser()
        result = parser.parse_file(str(temp_excel_file))
        assert isinstance(result, ParsedData)
        assert result.total_rows == 3
        assert result.total_columns == 3
        assert result.sheet_name is not None

    def test_parse_nonexistent_file(self) -> None:
        """Test parsing of non-existent file."""
        parser = FileParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.csv")

    def test_parse_invalid_format(self, tmp_path: Path) -> None:
        """Test parsing of invalid file format."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("some content")
        parser = FileParser()
        with pytest.raises(InvalidFormatError):
            parser.parse_file(str(invalid_file))

    def test_get_excel_sheet_names(self, temp_excel_file: Path) -> None:
        """Test Excel sheet name retrieval."""
        parser = FileParser()
        sheets = parser.get_sheet_names(str(temp_excel_file))
        assert isinstance(sheets, list)
        assert len(sheets) == 1
        assert sheets[0] == "Sheet1"

    def test_get_sheet_names_invalid_file(self, temp_csv_file: Path) -> None:
        """Test sheet name retrieval from non-Excel file."""
        parser = FileParser()
        with pytest.raises(InvalidFormatError):
            parser.get_sheet_names(str(temp_csv_file))

    def test_read_csv_chunks(self, temp_csv_file: Path) -> None:
        """Test chunked CSV reading."""
        parser = FileParser()
        chunks = list(parser.read_chunks(str(temp_csv_file), chunk_size=1))
        assert len(chunks) == 3
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
        assert all(len(chunk) == 1 for chunk in chunks)

    def test_parse_with_custom_options(self, temp_csv_file: Path) -> None:
        """Test parsing with custom options."""
        options = ParseOptions(
            encoding="utf-8",
            chunk_size=500,
            na_values=["NA", "N/A"],
            parse_dates=[],
            date_format="%Y-%m-%d",
        )
        parser = FileParser()
        result = parser.parse_file(str(temp_csv_file), options)
        assert isinstance(result, ParsedData)
        assert result.total_rows == 3

    def test_parse_corrupt_excel(self, corrupt_excel_file: Path) -> None:
        """Test handling of corrupt Excel files."""
        parser = FileParser()
        with pytest.raises(InvalidStructureError):
            parser.parse_file(str(corrupt_excel_file))

    def test_parse_excel_specific_sheet(self, tmp_path: Path) -> None:
        """Test parsing specific Excel sheet."""
        file_path = tmp_path / "multi_sheet.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            pd.DataFrame({"id": [1]}).to_excel(writer, "Sheet1")
            pd.DataFrame({"id": [2]}).to_excel(writer, "Sheet2")

        parser = FileParser()
        options = ParseOptions(sheet_name="Sheet2")
        result = parser.parse_file(str(file_path), options)
        assert result.sheet_name == "Sheet2"
        assert result.data["id"].iloc[0] == 2

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Test parsing of empty files."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        parser = FileParser()
        with pytest.raises(InvalidStructureError):
            parser.parse_file(str(empty_file))

    @pytest.mark.parametrize("chunk_size", [1, 2, 5])
    def test_chunk_sizes(self, chunk_size: int, temp_csv_file: Path) -> None:
        """Test different chunk sizes."""
        parser = FileParser()
        chunks = list(parser.read_chunks(str(temp_csv_file), chunk_size))
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 3  # Total rows in sample file
