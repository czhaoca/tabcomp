"""
Unit tests for the FileValidator component.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from tabcomp.file_handler.exceptions import (
    FileNotFoundError,
    FilePermissionError,
    FileLockError,
    InvalidFormatError,
    InvalidStructureError,
)
from tabcomp.file_handler.validator import FileValidator, ValidationResult


class TestFileValidator:
    """Test suite for FileValidator class."""

    def test_init_with_custom_size(self) -> None:
        """Test initialization with custom max file size."""
        custom_size = 1000
        validator = FileValidator(max_file_size=custom_size)
        assert validator.max_file_size == custom_size

    def test_validate_file_success(self, temp_csv_file: Path) -> None:
        """Test successful file validation."""
        validator = FileValidator()
        result = validator.validate_file(str(temp_csv_file))
        assert result.is_valid
        assert not result.errors
        assert not result.warnings

    def test_validate_nonexistent_file(self) -> None:
        """Test validation of non-existent file."""
        validator = FileValidator()
        with pytest.raises(FileNotFoundError) as exc_info:
            validator.validate_file("nonexistent.csv")
        assert "File not found" in str(exc_info.value)

    @patch("os.path.getsize")
    def test_validate_file_size_limit(
        self, mock_getsize: Mock, temp_csv_file: Path
    ) -> None:
        """Test validation of file size limits."""
        mock_getsize.return_value = FileValidator.MAX_FILE_SIZE + 1
        validator = FileValidator()
        with pytest.raises(InvalidFormatError) as exc_info:
            validator.validate_file(str(temp_csv_file))
        assert "size" in str(exc_info.value)

    def test_validate_invalid_extension(self, tmp_path: Path) -> None:
        """Test validation of invalid file extensions."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("some content")
        validator = FileValidator()
        with pytest.raises(InvalidFormatError) as exc_info:
            validator.validate_file(str(invalid_file))
        assert "Invalid file format" in str(exc_info.value)

    @patch("builtins.open", side_effect=PermissionError)
    def test_validate_permission_denied(
        self, mock_open: Mock, temp_csv_file: Path
    ) -> None:
        """Test validation when file access is denied."""
        validator = FileValidator()
        with pytest.raises(FilePermissionError) as exc_info:
            validator.validate_file(str(temp_csv_file))
        assert "Permission denied" in str(exc_info.value)

    @patch("builtins.open", side_effect=OSError)
    def test_validate_file_locked(self, mock_open: Mock, temp_csv_file: Path) -> None:
        """Test validation when file is locked."""
        validator = FileValidator()
        with pytest.raises(FileLockError) as exc_info:
            validator.validate_file(str(temp_csv_file))
        assert "File is locked" in str(exc_info.value)

    def test_validate_empty_file(self, tmp_path: Path) -> None:
        """Test validation of empty files."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        validator = FileValidator()
        with pytest.raises(InvalidStructureError) as exc_info:
            validator.validate_file(str(empty_file))
        assert "invalid" in str(exc_info.value).lower()

    def test_validate_missing_required_columns(self, tmp_path: Path) -> None:
        """Test validation of files missing required columns."""
        file_path = tmp_path / "invalid.csv"
        pd.DataFrame({"column1": [1, 2], "column2": ["a", "b"]}).to_csv(
            file_path, index=False
        )

        validator = FileValidator()
        with pytest.raises(InvalidStructureError) as exc_info:
            validator.validate_file(str(file_path))
        assert "missing" in str(exc_info.value).lower()

    def test_validate_excel_multiple_sheets(self, tmp_path: Path) -> None:
        """Test validation of Excel files with multiple sheets."""
        file_path = tmp_path / "multi_sheet.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            pd.DataFrame({"id": [1], "name": ["test"], "value": [100]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )
            pd.DataFrame({"id": [2], "name": ["test2"], "value": [200]}).to_excel(
                writer, sheet_name="Sheet2", index=False
            )

        validator = FileValidator()
        result = validator.validate_file(str(file_path))
        assert result.is_valid

    @pytest.mark.parametrize("extension", [".xlsx", ".xls", ".csv"])
    def test_validate_all_supported_formats(
        self, extension: str, tmp_path: Path
    ) -> None:
        """Test validation of all supported file formats."""
        file_path = tmp_path / f"test{extension}"
        df = pd.DataFrame({"id": [1], "name": ["test"], "value": [100]})

        if extension == ".csv":
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        validator = FileValidator()
        result = validator.validate_file(str(file_path))
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
