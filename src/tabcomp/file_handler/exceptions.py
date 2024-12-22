"""
Custom exceptions for the TabComp File Handler module.

This module defines a hierarchy of exceptions specific to file handling operations,
including validation, parsing, and encoding detection.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class FileHandlerError(Exception):
    """Base exception class for all file handler errors."""

    def __init__(
        self,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
    ):
        """
        Initialize the base exception.

        Args:
            message: User-friendly error message
            code: Unique error identifier code
            details: Additional technical details about the error
            suggested_action: Recommended action to resolve the error
        """
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.suggested_action = suggested_action
        self.timestamp = datetime.now()


class FileAccessError(FileHandlerError):
    """Raised when there are issues accessing the file."""

    def __init__(
        self,
        message: str,
        code: str = "FA-001",
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message, code, details, suggested_action)


class FileNotFoundError(FileAccessError):
    """Raised when the specified file does not exist."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File not found: {filepath}",
            code="FA-001",
            details=details,
            suggested_action="Please verify the file path and ensure the file exists.",
        )


class FilePermissionError(FileAccessError):
    """Raised when there are insufficient permissions to access the file."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Permission denied: {filepath}",
            code="FA-002",
            details=details,
            suggested_action="Please check file permissions and ensure you have access rights.",
        )


class FileLockError(FileAccessError):
    """Raised when the file is locked by another process."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File is locked: {filepath}",
            code="FA-003",
            details=details,
            suggested_action="Please close any applications that might be using the file.",
        )


class FormatError(FileHandlerError):
    """Base class for file format related errors."""

    def __init__(
        self,
        message: str,
        code: str = "FMT-001",
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message, code, details, suggested_action)


class InvalidFormatError(FormatError):
    """Raised when the file format is not supported or invalid."""

    def __init__(
        self, filepath: str, format: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Invalid file format: {format} for file {filepath}",
            code="FMT-001",
            details=details,
            suggested_action="Please ensure the file is a valid Excel (.xlsx, .xls) or CSV file.",
        )


class CorruptFileError(FormatError):
    """Raised when the file is corrupted and cannot be processed."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File is corrupted: {filepath}",
            code="FMT-002",
            details=details,
            suggested_action="The file appears to be corrupted. Please verify the file integrity.",
        )


class InvalidStructureError(FormatError):
    """Raised when the file structure does not match expected format."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Invalid file structure: {filepath}",
            code="FMT-003",
            details=details,
            suggested_action="Please ensure the file has the correct structure with valid headers and data.",
        )


class EncodingError(FileHandlerError):
    """Base class for encoding related errors."""

    def __init__(
        self,
        message: str,
        code: str = "ENC-001",
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message, code, details, suggested_action)


class UnknownEncodingError(EncodingError):
    """Raised when file encoding cannot be detected."""

    def __init__(self, filepath: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Unable to detect encoding for file: {filepath}",
            code="ENC-001",
            details=details,
            suggested_action="Please specify the file encoding manually or save the file with UTF-8 encoding.",
        )


class EncodingMismatchError(EncodingError):
    """Raised when detected encoding doesn't match specified encoding."""

    def __init__(
        self,
        filepath: str,
        expected: str,
        detected: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Encoding mismatch in {filepath}: expected {expected}, detected {detected}",
            code="ENC-002",
            details=details,
            suggested_action=f"Please verify the file encoding or try opening with {detected} encoding.",
        )


class DecodeError(EncodingError):
    """Raised when file content cannot be decoded with the specified encoding."""

    def __init__(
        self, filepath: str, encoding: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Failed to decode file {filepath} with encoding {encoding}",
            code="ENC-003",
            details=details,
            suggested_action="Please verify the file encoding or try saving the file with UTF-8 encoding.",
        )
