"""
File encoding detection and handling module for TabComp.

This module provides functionality for detecting, validating, and handling
different file encodings, with support for both CSV and Excel files.
"""

import os
import chardet
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .exceptions import UnknownEncodingError, EncodingMismatchError, DecodeError


@dataclass
class EncodingInfo:
    """Contains encoding detection results."""

    encoding: str
    confidence: float
    language: Optional[str] = None


class EncodingDetector:
    """Handles file encoding detection and validation."""

    # Common encodings to check first
    COMMON_ENCODINGS = [
        "utf-8",
        "ascii",
        "iso-8859-1",
        "windows-1252",
        "utf-16",
        "utf-32",
    ]

    # Minimum confidence threshold for encoding detection
    MIN_CONFIDENCE = 0.8

    # Maximum bytes to read for detection
    SAMPLE_SIZE = 50 * 1024  # 50KB

    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        """
        Initialize the EncodingDetector.

        Args:
            min_confidence: Minimum confidence threshold for encoding detection.
                          Defaults to MIN_CONFIDENCE class constant.
        """
        self.min_confidence = min_confidence

    def detect_encoding(self, file_path: str) -> EncodingInfo:
        """
        Detects the encoding of a file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            EncodingInfo containing detected encoding and confidence level

        Raises:
            UnknownEncodingError: If encoding cannot be detected with sufficient confidence
        """
        # Try common encodings first for better performance
        encoding_info = self._try_common_encodings(file_path)
        if encoding_info:
            return encoding_info

        # Fall back to chardet for more thorough detection
        encoding_info = self._detect_with_chardet(file_path)
        if encoding_info and encoding_info.confidence >= self.min_confidence:
            return encoding_info

        raise UnknownEncodingError(
            file_path,
            details={
                "attempted_encodings": self.COMMON_ENCODINGS,
                "chardet_result": encoding_info.__dict__ if encoding_info else None,
            },
        )

    def validate_encoding(self, file_path: str, expected_encoding: str) -> None:
        """
        Validates if file content matches expected encoding.

        Args:
            file_path: Path to the file to validate
            expected_encoding: Expected encoding of the file

        Raises:
            EncodingMismatchError: If detected encoding doesn't match expected encoding
            DecodeError: If file cannot be decoded with specified encoding
        """
        try:
            detected = self.detect_encoding(file_path)
            if detected.encoding.lower() != expected_encoding.lower():
                raise EncodingMismatchError(
                    file_path,
                    expected_encoding,
                    detected.encoding,
                    details={"detection_confidence": detected.confidence},
                )

            # Verify we can actually decode the content
            self._verify_decoding(file_path, expected_encoding)

        except UnknownEncodingError:
            # Convert to DecodeError if we can't even detect the encoding
            raise DecodeError(
                file_path,
                expected_encoding,
                details={"error": "Unable to detect file encoding"},
            )

    def get_file_bom(self, file_path: str) -> Optional[bytes]:
        """
        Detects the Byte Order Mark (BOM) of a file if present.

        Args:
            file_path: Path to the file to check

        Returns:
            Bytes object containing the BOM if found, None otherwise
        """
        bom_marks = {
            (0xEF, 0xBB, 0xBF): "utf-8-sig",
            (0xFE, 0xFF): "utf-16be",
            (0xFF, 0xFE): "utf-16le",
            (0x00, 0x00, 0xFE, 0xFF): "utf-32be",
            (0xFF, 0xFE, 0x00, 0x00): "utf-32le",
        }

        with open(file_path, "rb") as f:
            raw = f.read(4)  # Read enough for longest BOM
            for bom_bytes, encoding in bom_marks.items():
                if raw.startswith(bytes(bom_bytes)):
                    return bytes(bom_bytes)
        return None

    def _try_common_encodings(self, file_path: str) -> Optional[EncodingInfo]:
        """
        Attempts to decode file with common encodings.

        Args:
            file_path: Path to the file to check

        Returns:
            EncodingInfo if successful match found, None otherwise
        """
        with open(file_path, "rb") as f:
            raw_content = f.read(self.SAMPLE_SIZE)

        for encoding in self.COMMON_ENCODINGS:
            try:
                raw_content.decode(encoding)
                return EncodingInfo(
                    encoding=encoding,
                    confidence=1.0,  # We're certain if decode succeeds
                )
            except UnicodeError:
                continue
        return None

    def _detect_with_chardet(self, file_path: str) -> Optional[EncodingInfo]:
        """
        Uses chardet library for encoding detection.

        Args:
            file_path: Path to the file to analyze

        Returns:
            EncodingInfo containing chardet detection results
        """
        with open(file_path, "rb") as f:
            raw_content = f.read(self.SAMPLE_SIZE)

        result = chardet.detect(raw_content)
        if result["encoding"]:
            return EncodingInfo(
                encoding=result["encoding"],
                confidence=result["confidence"],
                language=result.get("language"),
            )
        return None

    def _verify_decoding(self, file_path: str, encoding: str) -> None:
        """
        Verifies that file content can be decoded with specified encoding.

        Args:
            file_path: Path to the file to verify
            encoding: Encoding to test

        Raises:
            DecodeError: If content cannot be decoded with specified encoding
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read(self.SAMPLE_SIZE)
                content.decode(encoding)
        except UnicodeError as e:
            raise DecodeError(file_path, encoding, details={"error": str(e)})


class EncodingConverter:
    """Handles conversion between different encodings."""

    def convert_encoding(
        self,
        file_path: str,
        target_encoding: str,
        source_encoding: Optional[str] = None,
    ) -> str:
        """
        Converts file content to target encoding.

        Args:
            file_path: Path to the source file
            target_encoding: Desired output encoding
            source_encoding: Optional source encoding (will be detected if not provided)

        Returns:
            Path to the converted file

        Raises:
            DecodeError: If content cannot be decoded from source encoding
            EncodingMismatchError: If detected encoding doesn't match provided source encoding
        """
        detector = EncodingDetector()

        # Detect or verify source encoding
        if source_encoding:
            detector.validate_encoding(file_path, source_encoding)
        else:
            source_encoding = detector.detect_encoding(file_path).encoding

        # Create output file path
        base_path = os.path.splitext(file_path)[0]
        output_path = f"{base_path}_{target_encoding}{os.path.splitext(file_path)[1]}"

        # Perform conversion
        try:
            with open(file_path, "r", encoding=source_encoding) as source:
                content = source.read()

            with open(output_path, "w", encoding=target_encoding) as target:
                target.write(content)

            return output_path

        except UnicodeError as e:
            raise DecodeError(file_path, source_encoding, details={"error": str(e)})
