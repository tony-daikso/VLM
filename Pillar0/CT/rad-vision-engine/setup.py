#!/usr/bin/env python
"""Setup script for rad-vision-engine package."""

from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]


def _collect_config_files():
    """Collect YAML configs so they are included in wheels and sdists."""
    config_dir = Path("configs")
    if not config_dir.exists():
        return []

    dir_to_files = {}
    for yaml_file in config_dir.rglob("*.yaml"):
        rel_dir = yaml_file.parent.relative_to(config_dir)
        install_dir = Path("configs") / rel_dir
        dir_to_files.setdefault(str(install_dir), []).append(str(yaml_file))

    # setuptools expects a list of (directory, [files])
    return [
        (directory, sorted(files))
        for directory, files in sorted(dir_to_files.items(), key=lambda item: item[0])
    ]


config_data_files = _collect_config_files()

setup(
    name="rad-vision-engine",
    version="1.0.0",
    author="Your Team",
    author_email="yala@berkeley.edu",
    description="A high-performance medical image processing engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YalaLab/rad-vision-engine",
    packages=find_packages() + ["rve"],
    package_data={
        "vision_engine": ["**/*.yaml"],
        "rve": ["*.py"],
    },
    include_package_data=True,
    data_files=config_data_files,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vision-engine=vision_engine.cli.main:main",
            "rve=vision_engine.cli.main:main",  # Alias for convenience
        ],
    },
)
