"""
File validation module for TabComp.

This module handles validation of input files, including size checks,
format verification, and structure validation for Excel and CSV files.
"""

import os
import pathlib
from typing import List, Optional, Tuple, Set
import pandas as pd
from dataclasses import dataclass

from .exceptions import (
    FileNotFoundError,
    FilePermissionError,
    FileLockError,
    InvalidFormatError,
    InvalidStructureError,
)


@dataclass
class ValidationResult:
    """Contains the results of file validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


class FileValidator:
    """Validates input files for size, format, and structure."""

    # Class constants
    MAX_FILE_SIZE = 500_000_000  # 500MB
    ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
    REQUIRED_COLUMNS = {"id", "name", "value"}  # Example required columns

    def __init__(self, max_file_size: Optional[int] = None):
        """
        Initialize the FileValidator.

        Args:
            max_file_size: Optional custom maximum file size in bytes.
                          Defaults to MAX_FILE_SIZE class constant.
        """
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE

    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Performs comprehensive file validation.

        Args:
            file_path: Path to the file to validate

        Returns:
            ValidationResult containing validation status and any errors/warnings

        Raises:
            FileNotFoundError: If the file does not exist
            FilePermissionError: If the file cannot be accessed
            InvalidFormatError: If the file format is not supported
        """
        errors = []
        warnings = []

        # Check basic file access and format
        try:
            self._validate_file_exists(file_path)
            self._validate_file_access(file_path)
            self._validate_file_format(file_path)
            self._validate_file_size(file_path)
        except (FileNotFoundError, FilePermissionError, InvalidFormatError) as e:
            return ValidationResult(is_valid=False, errors=[str(e)], warnings=[])

        # Validate file structure based on format
        try:
            self._validate_file_structure(file_path)
        except InvalidStructureError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected error during structure validation: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _validate_file_exists(self, file_path: str) -> None:
        """
        Validates that the file exists.

        Args:
            file_path: Path to the file to validate

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

    def _validate_file_access(self, file_path: str) -> None:
        """
        Validates that the file can be accessed with required permissions.

        Args:
            file_path: Path to the file to validate

        Raises:
            FilePermissionError: If the file cannot be accessed
            FileLockError: If the file is locked by another process
        """
        try:
            with open(file_path, "rb"):
                pass
        except PermissionError:
            raise FilePermissionError(file_path)
        except OSError:
            raise FileLockError(file_path)

    def _validate_file_format(self, file_path: str) -> None:
        """
        Validates that the file has an allowed extension.

        Args:
            file_path: Path to the file to validate

        Raises:
            InvalidFormatError: If the file format is not supported
        """
        file_extension = pathlib.Path(file_path).suffix.lower()
        if file_extension not in self.ALLOWED_EXTENSIONS:
            raise InvalidFormatError(
                file_path,
                file_extension,
                details={"allowed_extensions": list(self.ALLOWED_EXTENSIONS)},
            )

    def _validate_file_size(self, file_path: str) -> None:
        """
        Validates that the file size is within allowed limits.

        Args:
            file_path: Path to the file to validate

        Raises:
            InvalidFormatError: If the file size exceeds the maximum allowed size
        """
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise InvalidFormatError(
                file_path,
                "size",
                details={"file_size": file_size, "max_size": self.max_file_size},
            )

    def _validate_file_structure(self, file_path: str) -> None:
        """
        Validates the internal structure of the file.

        Args:
            file_path: Path to the file to validate

        Raises:
            InvalidStructureError: If the file structure is invalid
        """
        file_extension = pathlib.Path(file_path).suffix.lower()

        if file_extension == ".csv":
            self._validate_csv_structure(file_path)
        else:  # Excel files
            self._validate_excel_structure(file_path)

    def _validate_csv_structure(self, file_path: str) -> None:
        """
        Validates the structure of a CSV file.

        Args:
            file_path: Path to the CSV file

        Raises:
            InvalidStructureError: If the CSV structure is invalid
        """
        try:
            # Read only the header row for validation
            df = pd.read_csv(file_path, nrows=0)
            columns = set(df.columns)

            missing_columns = self.REQUIRED_COLUMNS - columns
            if missing_columns:
                raise InvalidStructureError(
                    file_path,
                    details={
                        "missing_columns": list(missing_columns),
                        "required_columns": list(self.REQUIRED_COLUMNS),
                    },
                )
        except pd.errors.EmptyDataError:
            raise InvalidStructureError(file_path, details={"error": "File is empty"})
        except Exception as e:
            raise InvalidStructureError(file_path, details={"error": str(e)})

    def _validate_excel_structure(self, file_path: str) -> None:
        """
        Validates the structure of an Excel file.

        Args:
            file_path: Path to the Excel file

        Raises:
            InvalidStructureError: If the Excel structure is invalid
        """
        try:
            # Read only the header row from the first sheet
            df = pd.read_excel(file_path, nrows=0)
            columns = set(df.columns)

            missing_columns = self.REQUIRED_COLUMNS - columns
            if missing_columns:
                raise InvalidStructureError(
                    file_path,
                    details={
                        "missing_columns": list(missing_columns),
                        "required_columns": list(self.REQUIRED_COLUMNS),
                    },
                )
        except Exception as e:
            raise InvalidStructureError(file_path, details={"error": str(e)})
