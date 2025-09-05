#!/usr/bin/env python3
"""
Setup script for Real-Time Object Detection & Tracking package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="real-time-object-detection-tracking",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-Time Object Detection & Tracking System with YOLOv8 and OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/real-time-object-detection-tracking",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "real-time-detection=main:main",
            "rt-detection=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "computer-vision",
        "object-detection",
        "object-tracking",
        "yolo",
        "yolov8",
        "opencv",
        "real-time",
        "surveillance",
        "bytetrack",
        "deep-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/real-time-object-detection-tracking/issues",
        "Source": "https://github.com/yourusername/real-time-object-detection-tracking",
        "Documentation": "https://github.com/yourusername/real-time-object-detection-tracking#readme",
    },
)
