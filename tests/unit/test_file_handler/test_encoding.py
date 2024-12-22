"""
Unit tests for the EncodingDetector component.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tabcomp.file_handler.exceptions import (
    UnknownEncodingError,
    EncodingMismatchError,
    DecodeError,
)
from tabcomp.file_handler.encoding import (
    EncodingDetector,
    EncodingInfo,
    EncodingConverter,
)


class TestEncodingDetector:
    """Test suite for EncodingDetector class."""

    def test_detect_utf8_encoding(self, temp_csv_file: Path) -> None:
        """Test UTF-8 encoding detection."""
        detector = EncodingDetector()
        result = detector.detect_encoding(str(temp_csv_file))
        assert isinstance(result, EncodingInfo)
        assert result.encoding.lower() in ("utf-8", "ascii")
        assert result.confidence > 0.8

    def test_detect_utf16_encoding(self, utf16_file: Path) -> None:
        """Test UTF-16 encoding detection."""
        detector = EncodingDetector()
        result = detector.detect_encoding(str(utf16_file))
        assert result.encoding.lower().startswith("utf-16")

    def test_detect_unknown_encoding(self, tmp_path: Path) -> None:
        """Test handling of unknown encodings."""
        file_path = tmp_path / "unknown.csv"
        with open(file_path, "wb") as f:
            f.write(b"\xFF\xFE\xFF\xFE")  # Invalid encoding

        detector = EncodingDetector()
        with pytest.raises(UnknownEncodingError):
            detector.detect_encoding(str(file_path))

    def test_validate_encoding_match(self, temp_csv_file: Path) -> None:
        """Test successful encoding validation."""
        detector = EncodingDetector()
        # Should not raise an exception
        detector.validate_encoding(str(temp_csv_file), "utf-8")

    def test_validate_encoding_mismatch(self, utf16_file: Path) -> None:
        """Test encoding mismatch detection."""
        detector = EncodingDetector()
        with pytest.raises(EncodingMismatchError):
            detector.validate_encoding(str(utf16_file), "utf-8")

    def test_get_file_bom_utf8(self, tmp_path: Path) -> None:
        """Test BOM detection for UTF-8."""
        file_path = tmp_path / "bom.csv"
        with open(file_path, "wb") as f:
            f.write(b"\xEF\xBB\xBF")  # UTF-8 BOM
            f.write(b"test content")

        detector = EncodingDetector()
        bom = detector.get_file_bom(str(file_path))
        assert bom == b"\xEF\xBB\xBF"

    def test_get_file_bom_none(self, temp_csv_file: Path) -> None:
        """Test BOM detection when no BOM present."""
        detector = EncodingDetector()
        bom = detector.get_file_bom(str(temp_csv_file))
        assert bom is None

    @pytest.mark.parametrize("confidence", [0.7, 0.9])
    def test_minimum_confidence_threshold(
        self, confidence: float, temp_csv_file: Path
    ) -> None:
        """Test minimum confidence threshold handling."""
        detector = EncodingDetector(min_confidence=confidence)
        result = detector.detect_encoding(str(temp_csv_file))
        assert result.confidence >= confidence


class TestEncodingConverter:
    """Test suite for EncodingConverter class."""

    def test_convert_to_utf8(self, utf16_file: Path) -> None:
        """Test conversion to UTF-8."""
        converter = EncodingConverter()
        output_path = converter.convert_encoding(str(utf16_file), "utf-8", "utf-16")

        # Verify conversion
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "id,name,value" in content

    def test_convert_with_auto_detection(self, utf16_file: Path) -> None:
        """Test conversion with automatic source encoding detection."""
        converter = EncodingConverter()
        output_path = converter.convert_encoding(str(utf16_file), "utf-8")

        # Verify conversion
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "id,name,value" in content

    def test_convert_invalid_target_encoding(self, temp_csv_file: Path) -> None:
        """Test handling of invalid target encoding."""
        converter = EncodingConverter()
        with pytest.raises(LookupError):
            converter.convert_encoding(str(temp_csv_file), "invalid-encoding")
