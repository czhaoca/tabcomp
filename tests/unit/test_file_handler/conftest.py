"""Shared fixtures for file handler tests."""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_csv_content() -> str:
    return "id,name,value\n1,Test,100\n2,Example,200"


@pytest.fixture
def sample_excel_content() -> bytes:
    df = pd.DataFrame({"id": [1, 2], "name": ["Test", "Example"], "value": [100, 200]})
    return df.to_excel(None, index=False).getvalue()


@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_csv_content: str) -> Path:
    file_path = tmp_path / "test.csv"
    file_path.write_text(sample_csv_content)
    return file_path


@pytest.fixture
def temp_excel_file(tmp_path: Path, sample_excel_content: bytes) -> Path:
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(sample_excel_content)
    return file_path
