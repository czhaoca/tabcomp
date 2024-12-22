"""
Integration tests for the File Handler module.
"""

import pytest
import pandas as pd
from pathlib import Path
import time
import psutil
import os

from tabcomp.file_handler.validator import FileValidator
from tabcomp.file_handler.encoding import EncodingDetector, EncodingConverter
from tabcomp.file_handler.parser import FileParser, ParseOptions


class TestFileHandlerIntegration:
    """Integration test suite for File Handler module."""

    def test_full_processing_pipeline(self, temp_csv_file: Path) -> None:
        """Test complete file processing pipeline."""
        # 1. Validate file
        validator = FileValidator()
        validation_result = validator.validate_file(str(temp_csv_file))
        assert validation_result.is_valid

        # 2. Detect encoding
        detector = EncodingDetector()
        encoding_info = detector.detect_encoding(str(temp_csv_file))
        assert encoding_info.encoding

        # 3. Parse file
        parser = FileParser(encoding_detector=detector)
        options = ParseOptions(encoding=encoding_info.encoding)
        parsed_data = parser.parse_file(str(temp_csv_file), options)

        # Verify results
        assert parsed_data.total_rows == 3
        assert parsed_data.total_columns == 3
        assert all(col in parsed_data.column_types for col in ["id", "name", "value"])

    def test_large_file_processing(self, large_csv_file: Path) -> None:
        """Test processing of large files."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Process in chunks
        parser = FileParser()
        chunk_count = 0
        total_rows = 0

        for chunk in parser.read_chunks(str(large_csv_file), chunk_size=10000):
            chunk_count += 1
            total_rows += len(chunk)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        # Performance assertions
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        file_size = os.path.getsize(large_csv_file)

        assert processing_time < 3.0  # Should process within 3 seconds
        assert (
            memory_used < file_size * 2
        )  # Memory usage should be less than 2x file size
        assert total_rows == 100_000  # Verify all rows were processed

    def test_multi_sheet_excel_processing(self, tmp_path: Path) -> None:
        """Test processing of multi-sheet Excel files."""
        # Create test file with multiple sheets
        file_path = tmp_path / "multi_sheet.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            pd.DataFrame({"id": [1], "name": ["test1"], "value": [100]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )
            pd.DataFrame({"id": [2], "name": ["test2"], "value": [200]}).to_excel(
                writer, sheet_name="Sheet2", index=False
            )

        # Process each sheet
        parser = FileParser()
        sheet_names = parser.get_sheet_names(str(file_path))

        for sheet in sheet_names:
            options = ParseOptions(sheet_name=sheet)
            result = parser.parse_file(str(file_path), options)
            assert result.total_rows == 1
            assert result.sheet_name == sheet

    def test_error_handling_pipeline(self, corrupt_excel_file: Path) -> None:
        """Test error handling throughout the processing pipeline."""
        validator = FileValidator()
        parser = FileParser()

        # Should fail validation
        with pytest.raises(InvalidStructureError):
            validator.validate_file(str(corrupt_excel_file))

        # Should fail parsing
        with pytest.raises(InvalidStructureError):
            parser.parse_file(str(corrupt_excel_file))

    def test_encoding_conversion_pipeline(self, utf16_file: Path) -> None:
        """Test encoding detection and conversion pipeline."""
        # 1. Detect original encoding
        detector = EncodingDetector()
        encoding_info = detector.detect_encoding(str(utf16_file))
        assert encoding_info.encoding.lower().startswith("utf-16")

        # 2. Convert to UTF-8
        converter = EncodingConverter()
        converted_path = converter.convert_encoding(
            str(utf16_file), "utf-8", encoding_info.encoding
        )

        # 3. Verify conversion
        new_encoding = detector.detect_encoding(converted_path)
        assert new_encoding.encoding.lower() == "utf-8"

        # 4. Parse converted file
        parser = FileParser()
        result = parser.parse_file(converted_path)
        assert result.total_rows > 0

    @pytest.mark.performance
    def test_memory_efficiency(self, large_csv_file: Path) -> None:
        """Test memory usage during file processing."""
        chunk_size = 10000
        parser = FileParser()

        # Monitor memory usage during chunk processing
        peak_memory = 0
        for chunk in parser.read_chunks(str(large_csv_file), chunk_size):
            current_memory = psutil.Process().memory_info().rss
            peak_memory = max(peak_memory, current_memory)

            # Basic processing to simulate real usage
            _ = chunk.describe()

        # Verify memory efficiency
        file_size = os.path.getsize(large_csv_file)
        assert peak_memory < file_size * 2  # Memory usage should be bounded

    @pytest.mark.performance
    def test_processing_speed(self, large_csv_file: Path) -> None:
        """Test processing speed requirements."""
        parser = FileParser()
        start_time = time.time()

        # Process entire file
        total_rows = sum(
            len(chunk) for chunk in parser.read_chunks(str(large_csv_file))
        )

        processing_time = time.time() - start_time
        rows_per_second = total_rows / processing_time

        assert rows_per_second > 50000  # Should process at least 50k rows per second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
