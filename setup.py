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
        "chardet",  # for encoding detection
    ],
    python_requires=">=3.11",
)
