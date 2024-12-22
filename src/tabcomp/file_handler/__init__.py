"""
TabComp File Handler module.

This module provides functionality for handling, validating, and parsing
Excel and CSV files with support for different encodings and large file processing.
"""

from typing import List

from .validator import FileValidator, ValidationResult
from .encoding import EncodingDetector, EncodingConverter, EncodingInfo
from .parser import FileParser, ParseOptions, ParsedData
from .exceptions import (
    FileHandlerError,
    FileAccessError,
    FileNotFoundError,
    FilePermissionError,
    FileLockError,
    FormatError,
    InvalidFormatError,
    CorruptFileError,
    InvalidStructureError,
    EncodingError,
    UnknownEncodingError,
    EncodingMismatchError,
    DecodeError,
)

# Module version
__version__ = "1.0.0"

# Module metadata
__author__ = "TabComp Team"
__description__ = "File handling and parsing for Excel and CSV files"

# Public API
__all__: List[str] = [
    # Main classes
    "FileValidator",
    "EncodingDetector",
    "EncodingConverter",
    "FileParser",
    # Data classes
    "ValidationResult",
    "EncodingInfo",
    "ParseOptions",
    "ParsedData",
    # Exceptions
    "FileHandlerError",
    "FileAccessError",
    "FileNotFoundError",
    "FilePermissionError",
    "FileLockError",
    "FormatError",
    "InvalidFormatError",
    "CorruptFileError",
    "InvalidStructureError",
    "EncodingError",
    "UnknownEncodingError",
    "EncodingMismatchError",
    "DecodeError",
]


def get_version() -> str:
    """Return the module version."""
    return __version__


def get_supported_formats() -> List[str]:
    """Return list of supported file formats."""
    return [".xlsx", ".xls", ".csv"]
