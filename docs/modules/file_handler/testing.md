# File Handler Testing Guide

## Testing Strategy

### Unit Tests

- Located in `tests/unit/test_file_handler/`
- Test individual components in isolation
- Mock external dependencies
- Focus on edge cases and error handling

### Integration Tests

- Test complete workflows
- Verify component interactions
- Test real file operations
- Validate error propagation

### Performance Tests

- Measure processing speed
- Monitor memory usage
- Validate resource efficiency
- Ensure scalability

## Test Requirements

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements-test.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/unit/test_file_handler

# Run with coverage report
pytest tests/unit/test_file_handler --cov=tabcomp.file_handler --cov-report=html

# Run performance tests
pytest tests/unit/test_file_handler -m performance
```

### Coverage Requirements

- Minimum line coverage: 90%
- Minimum branch coverage: 85%
- Critical paths: 100%

### Performance Targets

- File loading: < 3 seconds for 100MB files
- Memory usage: < 2x file size
- Processing speed: > 50,000 rows/second

## Test Components

### FileValidator Tests

- File existence checks
- Format validation
- Size limit validation
- Structure validation
- Permission checks

### EncodingDetector Tests

- Encoding detection accuracy
- BOM detection
- Encoding conversion
- Error handling

### FileParser Tests

- CSV parsing
- Excel parsing
- Large file handling
- Multi-sheet support
- Error conditions

### Integration Tests

- End-to-end workflows
- Cross-component interaction
- Resource management
- Error propagation

## Writing New Tests

### Test Structure

```python
class TestComponent:
    """Test suite for component."""

    def test_specific_feature(self):
        """Test description."""
        # Arrange
        component = Component()

        # Act
        result = component.method()

        # Assert
        assert result.property == expected_value
```

### Best Practices

1. Use descriptive test names
2. Follow AAA pattern (Arrange, Act, Assert)
3. One assertion per test
4. Use appropriate fixtures
5. Handle cleanup properly

### Fixtures

- Use shared fixtures from conftest.py
- Create specific fixtures when needed
- Clean up resources properly
- Document fixture purpose

### Error Testing

- Test both success and failure paths
- Verify error messages
- Check error details
- Validate error codes

## Continuous Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: |
          python -m pip install -r requirements-test.txt
          pytest tests/unit/test_file_handler
```

### Performance Monitoring

- Track performance metrics
- Monitor resource usage
- Compare against baselines
- Alert on regressions

## Test Maintenance

### Regular Updates

- Update tests for new features
- Maintain coverage levels
- Update performance baselines
- Review test quality

### Documentation

- Keep README.md current
- Document new test cases
- Update coverage reports
- Maintain change logs
