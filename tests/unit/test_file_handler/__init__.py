"""
Test package for the TabComp File Handler module.

This package contains all tests for the file handler components:
- FileValidator
- EncodingDetector
- FileParser

Test files are organized by component, with shared fixtures in conftest.py
and integration tests in test_integration.py.
"""

import pytest
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Test package version
__version__ = "1.0.0"

# Minimum required pytest version
pytest_plugins = ["pytest-cov", "pytest-mock"]
