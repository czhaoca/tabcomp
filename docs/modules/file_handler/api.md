# File Handler API Reference

## Classes

### FileValidator

```python
class FileValidator:
    """Validates input files for size, format, and structure."""

    def __init__(self, max_file_size: Optional[int] = None) -> None:
        """
        Initialize FileValidator with optional size limit.

        Args:
            max_file_size: Maximum allowed file size in bytes (default: 500MB)
        """

    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Perform comprehensive file validation.

        Args:
            file_path: Path to the file to validate

        Returns:
            ValidationResult with validation status and any errors/warnings

        Raises:
            FileNotFoundError: If file doesn't exist
            FilePermissionError: If file cannot be accessed
            InvalidFormatError: If file format is not supported
        """
```

### EncodingDetector

```python
class EncodingDetector:
    """Handles file encoding detection and validation."""

    def __init__(self, min_confidence: float = 0.8) -> None:
        """
        Initialize EncodingDetector.

        Args:
            min_confidence: Minimum confidence threshold for detection
        """

    def detect_encoding(self, file_path: str) -> EncodingInfo:
        """
        Detect file encoding.

        Args:
            file_path: Path to file to analyze

        Returns:
            EncodingInfo with detected encoding and confidence level

        Raises:
            UnknownEncodingError: If encoding cannot be detected
        """

    def validate_encoding(self, file_path: str, expected_encoding: str) -> None:
        """
        Validate file encoding matches expected.

        Args:
            file_path: Path to file to validate
            expected_encoding: Expected encoding name

        Raises:
            EncodingMismatchError: If encoding doesn't match
            DecodeError: If content cannot be decoded
        """
```

### FileParser

```python
class FileParser:
    """Handles parsing of CSV and Excel files."""

    def __init__(self, encoding_detector: Optional[EncodingDetector] = None) -> None:
        """
        Initialize FileParser.

        Args:
            encoding_detector: Optional EncodingDetector instance
        """

    def parse_file(self, file_path: str, options: Optional[ParseOptions] = None) -> ParsedData:
        """
        Parse file and return structured data.

        Args:
            file_path: Path to file to parse
            options: Optional parsing configuration

        Returns:
            ParsedData containing data and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            InvalidFormatError: If format not supported
            InvalidStructureError: If structure invalid
            DecodeError: If encoding issues
        """

    def read_chunks(
        self,
        file_path: str,
        chunk_size: int = 10000,
        options: Optional[ParseOptions] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Read file in chunks for large files.

        Args:
            file_path: Path to file to read
            chunk_size: Number of rows per chunk
            options: Optional parsing configuration

        Yields:
            DataFrame chunks of specified size
        """
```

## Data Classes

### ValidationResult

```python
@dataclass
class ValidationResult:
    """Contains file validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
```

### EncodingInfo

```python
@dataclass
class EncodingInfo:
    """Contains encoding detection results."""
    encoding: str
    confidence: float
    language: Optional[str] = None
```

### ParseOptions

```python
@dataclass
class ParseOptions:
    """Configuration options for file parsing."""
    encoding: Optional[str] = None
    sheet_name: Optional[str] = None
    chunk_size: int = 10000
    na_values: List[str] = None
    parse_dates: List[str] = None
    dtype_backend: str = 'numpy'
    date_format: Optional[str] = None
```

### ParsedData

```python
@dataclass
class ParsedData:
    """Contains parsed file data and metadata."""
    data: pd.DataFrame
    sheet_name: Optional[str]
    total_rows: int
    total_columns: int
    column_types: Dict[str, str]
    null_counts: Dict[str, int]
```

## Exceptions

### Base Exceptions

```python
class FileHandlerError(Exception):
    """Base exception for all file handler errors."""

class FileAccessError(FileHandlerError):
    """Base for file access related errors."""

class FormatError(FileHandlerError):
    """Base for format related errors."""

class EncodingError(FileHandlerError):
    """Base for encoding related errors."""
```

### Specific Exceptions

```python
class FileNotFoundError(FileAccessError):
    """File does not exist."""

class FilePermissionError(FileAccessError):
    """Insufficient permissions to access file."""

class InvalidFormatError(FormatError):
    """File format not supported or invalid."""

class InvalidStructureError(FormatError):
    """File structure does not match requirements."""

class UnknownEncodingError(EncodingError):
    """File encoding cannot be detected."""

class EncodingMismatchError(EncodingError):
    """Detected encoding doesn't match expected."""

class DecodeError(EncodingError):
    """Content cannot be decoded with specified encoding."""
```

## Usage Examples

### Basic File Validation

```python
validator = FileValidator()
result = validator.validate_file("data.csv")
if result.is_valid:
    print("File is valid")
else:
    print("Validation errors:", result.errors)
```

### Encoding Detection

```python
detector = EncodingDetector()
encoding_info = detector.detect_encoding("data.csv")
print(f"File encoding: {encoding_info.encoding}")
print(f"Detection confidence: {encoding_info.confidence}")
```

### File Parsing

```python
parser = FileParser()
options = ParseOptions(
    encoding='utf-8',
    parse_dates=['date_column']
)
parsed_data = parser.parse_file("data.csv", options)
print(f"Total rows: {parsed_data.total_rows}")
```

### Large File Processing

```python
parser = FileParser()
for chunk in parser.read_chunks("large.csv", chunk_size=10000):
    process_chunk(chunk)  # Process each chunk separately
```
