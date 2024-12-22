# TabComp

An open-source table comparison tool that efficiently identifies and visualizes differences between Excel and CSV files. Built for data analysts and engineers who need reliable file comparison with support for multiple sheets, intelligent field matching, and detailed reporting.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/czhaoca/tabcomp/main.yml)

## 🚀 Features

- Compare Excel (.xlsx, .xls) and CSV files with intelligent field matching
- Support for multi-sheet Excel workbooks
- Automatic encoding detection and special character handling
- Configurable primary key selection
- Detailed difference reporting with customizable highlighting
- Comprehensive error handling and validation
- Cross-platform support (Windows, macOS)

## 🛠️ Installation

### Windows

```bash
# Download latest release
tabcomp-windows-x64.exe
```

### macOS

```bash
# Download latest release
tabcomp-macos-x64.app
```

### From Source

```bash
git clone https://github.com/czhaoca/tabcomp.git
cd tabcomp
pip install -r requirements.txt
python setup.py install
```

## 📖 Documentation

- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🔧 Configuration

Default settings in `config.py`:

```python
CONFIG = {
    'MAX_FILE_SIZE': 500_000_000,  # 500MB
    'PYTHON_VERSION': '>=3.11',     # Requires Python 3.11 or higher
    'HIGHLIGHT_COLORS': {
        'file1': '#ADD8E6',  # Light blue
        'file2': '#90EE90',  # Light green
        'diff': '#FFFF00'    # Yellow
    }
}
```

## 🤝 Contributing

TabComp is an open-source project and we warmly welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details. This means you can freely use, modify, and distribute this software, and any modifications must also be open source.

## 💡 Development Philosophy

TabComp follows a modular architecture with clear separation of concerns. Each component is designed to be independent and communicates through well-defined APIs, making it easy to maintain, test, and extend.

## ✨ Acknowledgments

This project was developed with the assistance of Claude, an AI assistant by Anthropic. The AI helped with architectural design, code structure, and documentation, demonstrating the potential of human-AI collaboration in open-source development.

## Performance Benchmarks

The following benchmarks are enforced through automated testing:

- File Loading: < 3 seconds for 100MB files
- Memory Usage: < 2x file size during processing
- Comparison Speed: > 100,000 rows/second
- UI Responsiveness: < 100ms for user interactions

Performance monitoring is automated through GitHub Actions:

```bash
# Run performance tests
pytest tests/performance/
```

## Error Handling

TabComp implements a comprehensive error handling system:

1. User-Facing Errors

   - Clear error messages with suggested actions
   - Non-technical descriptions for common issues
   - Progress indicators for long operations

2. Developer-Facing Errors
   - Detailed stack traces in logs
   - Error codes for automated handling
   - Context information for debugging

## Dependencies

Required Python packages (latest stable versions):

- pandas
- openpyxl
- tkinter
- xlsxwriter
- pytest (for testing)

## Automated Testing

Continuous Integration ensures compatibility with latest dependency versions:

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=tabcomp tests/
```
