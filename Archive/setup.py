"""Setup script for SV-NSDE package."""

from setuptools import setup, find_packages

setup(
    name="sv-nsde",
    version="0.1.0",
    description="Semantic Volatility-Modulated Neural SDE for Crisis Dynamics",
    author="Zirui CHEN",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
)
