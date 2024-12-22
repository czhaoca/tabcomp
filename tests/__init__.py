"""
TabComp test package initialization.

This module provides test configuration and shared utilities for all test modules.
It helps maintain consistent test setup across different test categories
(unit, integration, performance).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path to ensure imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test categories
TEST_CATEGORIES = {
    "unit": "tests/unit",
    "integration": "tests/integration",
    "performance": "tests/performance",
}

# Default test configuration
DEFAULT_TEST_CONFIG: Dict[str, Any] = {
    "max_file_size": 500_000_000,  # 500MB
    "chunk_size": 10000,
    "timeout": 300,  # seconds
    "performance": {
        "min_rows_per_second": 100_000,
        "max_memory_multiplier": 2.0,  # Maximum memory usage relative to file size
    },
}

# Test data configuration
TEST_DATA_CONFIG = {
    "csv_sample_size": 1000,
    "excel_sheet_count": 3,
    "default_encoding": "utf-8",
    "alternative_encodings": ["utf-16", "iso-8859-1", "windows-1252"],
}


def get_test_data_path() -> Path:
    """
    Get the path to the test data directory, creating it if it doesn't exist.

    Returns:
        Path to the test data directory
    """
    test_data_path = PROJECT_ROOT / "tests" / "data"
    test_data_path.mkdir(parents=True, exist_ok=True)
    return test_data_path


def get_test_category(test_path: str) -> str:
    """
    Determine the test category (unit, integration, performance) from the test path.

    Args:
        test_path: Path to the test file

    Returns:
        Test category name
    """
    for category, path in TEST_CATEGORIES.items():
        if path in test_path:
            return category
    return "unit"  # Default to unit test if no match


def get_test_config(category: str = "unit") -> Dict[str, Any]:
    """
    Get test configuration based on test category.

    Args:
        category: Test category (unit, integration, performance)

    Returns:
        Configuration dictionary for the specified test category
    """
    config = DEFAULT_TEST_CONFIG.copy()

    # Adjust configuration based on test category
    if category == "performance":
        # Use larger chunk size for performance tests
        config["chunk_size"] = 50000
        config["timeout"] = 600  # 10 minutes
    elif category == "integration":
        # Use moderate settings for integration tests
        config["chunk_size"] = 25000
        config["timeout"] = 450  # 7.5 minutes

    return config
