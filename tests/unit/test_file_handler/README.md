# File Handler Unit Tests

## Overview

This directory contains unit tests for the TabComp File Handler module components.

## Test Structure

```
tests/unit/test_file_handler/
├── __init__.py
├── conftest.py
├── test_validator.py
├── test_encoding.py
├── test_parser.py
└── test_integration.py
```

## Quick Start

```bash
# Run all tests
pytest tests/unit/test_file_handler

# Run specific test file
pytest tests/unit/test_file_handler/test_validator.py

# Run with coverage
pytest tests/unit/test_file_handler --cov=tabcomp.file_handler

# Run performance tests only
pytest tests/unit/test_file_handler -m performance
```

## Test Categories

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance tests for speed and memory usage

## Writing New Tests

1. Use appropriate fixtures from conftest.py
2. Follow naming convention: test\_\*.py
3. Include type hints
4. Add docstrings for test methods
5. Use appropriate markers for special tests
