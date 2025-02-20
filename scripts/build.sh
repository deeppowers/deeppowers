#!/bin/bash

# Check required tools
command -v cmake >/dev/null 2>&1 || { echo "cmake is required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }

# Install dependencies
echo "Installing dependencies..."
pip install -r ../requirements.txt

# Install GTest
echo "Installing GTest..."
if [ "$(uname)" == "Darwin" ]; then
    brew install googletest
else
    sudo apt-get install -y libgtest-dev
fi

# Create and enter build directory
echo "Creating build directory..."
mkdir -p ../build
cd ../build

# Configure CMake project
echo "Configuring CMake project..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
      ..

# Get number of CPU cores
if [ "$(uname)" == "Darwin" ]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=$(nproc)
fi

# Compile project
echo "Compiling project..."
cmake --build . -j${NUM_CORES}

# Back to project root
cd ..

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p src/core/{tokenizer,model,inference}/include
mkdir -p src/api/{cpp,python}/include
mkdir -p src/common/include
mkdir -p tests/{python,data,results}
mkdir -p examples

# Create example programs
echo "Creating example programs..."
cat > examples/basic_generation.cpp << 'EOL'
#include <iostream>
#include <deeppowers.hpp>

int main() {
    std::cout << "DeepPowers version: " << deeppowers::api::version() << std::endl;
    std::cout << "CUDA available: " << (deeppowers::api::cuda_available() ? "yes" : "no") << std::endl;
    return 0;
}
EOL

# Create example Python program
echo "Creating example Python program..."
cat > examples/python_example.py << 'EOL'
import deeppowers as dp

def main():
    print(f"DeepPowers version: {dp.__version__}")
    print(f"CUDA available: {dp.cuda_available()}")

if __name__ == "__main__":
    main()
EOL

# Create test data
echo "Creating test data..."
cat > tests/data/sample.txt << 'EOL'
This is a sample text for testing the tokenizer.
It contains multiple lines of text.
The tokenizer should be able to handle this correctly.
EOL

# Install Python packages
echo "Installing Python packages..."
cd src/api/python
pip install -e .

cd ../../..
echo "Build completed!" 