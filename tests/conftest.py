import os
import pytest
from pathlib import Path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test.csv"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("id,name,value\n1,test,100\n")
    return file_path


@pytest.fixture
def utf16_file(tmp_path):
    """Create a temporary UTF-16 encoded file for testing."""
    file_path = tmp_path / "test_utf16.csv"
    with open(file_path, "w", encoding="utf-16") as f:
        f.write("id,name,value\n1,test,100\n")
    return file_path


# tests/unit/test_file_handler/test_encoding.py
import pytest
from pathlib import Path
from tabcomp.file_handler.encoding import EncodingDetector, EncodingInfo
from tabcomp.file_handler.exceptions import UnknownEncodingError


def test_detect_utf8_encoding(temp_csv_file: Path) -> None:
    """Test UTF-8 encoding detection."""
    detector = EncodingDetector()
    result = detector.detect_encoding(str(temp_csv_file))
    assert isinstance(result, EncodingInfo)
    assert result.encoding.lower() in ("utf-8", "ascii")
    assert result.confidence > 0.8
