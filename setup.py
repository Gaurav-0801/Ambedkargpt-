"""Setup script for AmbedkarGPT - SEMRAG-Based RAG System."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with requirements_file.open(encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ambedkargpt",
    version="1.0.0",
    description="SEMRAG-Based RAG System for Dr. B.R. Ambedkar's works",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gaurav Pant",
    author_email="gauravpant.ind@gmail.com",
    url="https://github.com/yourusername/ambedkargpt",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ambedkargpt=src.pipeline.ambedkargpt:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)




