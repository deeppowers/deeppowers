from setuptools import setup, find_packages

# Read version from version.py
with open("deeppowers/version.py") as f:
    exec(f.read())

# Read README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deeppowers",
    version=__version__,
    description="High Performance Text Generation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepPowers Team",
    author_email="support@deeppowers.xyz",
    url="https://github.com/deeppowers/deeppowers",
    packages=find_packages(),
    package_data={
        "deeppowers": ["*.so", "*.pyd"],
    },
    install_requires=[
        "numpy>=1.20.0",
        "typing_extensions>=4.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning, text generation, nlp",
) 