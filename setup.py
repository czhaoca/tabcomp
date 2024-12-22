from setuptools import setup, find_packages

setup(
    name="tabcomp",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "openpyxl",
        "xlsxwriter",
        "pytest",
        "chardet",
        "psutil",  # Added psutil for system monitoring
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov",
            "black",
            "mypy",
        ],
        "test": [
            "pytest>=8.0.0",
            "pytest-cov",
            "psutil",
        ],
    },
    python_requires=">=3.10",
)
