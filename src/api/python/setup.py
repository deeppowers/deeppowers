from setuptools import setup, find_packages

# Read version from version.py
with open("deeppowers/version.py") as f:
    exec(f.read())

# Read README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deeppowers",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pybind11>=2.6.0",
        "torch>=1.8.0",  # Optional: only needed for PyTorch model support
    ],
    author="DeepPowers Team",
    author_email="support@deeppowers.xyz",
    description="High-performance inference engine for large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deeppowers/deeppowers",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 