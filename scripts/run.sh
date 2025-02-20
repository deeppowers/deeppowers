#!/bin/bash

# Check if build is complete
if [ ! -d "../build" ]; then
    echo "Build directory does not exist, running build script..."
    ./build.sh
fi

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../build/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/../build/lib  # For macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src/api/python

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import deeppowers as dp; print(f'CUDA available: {dp.cuda_available()}')" || {
    echo "Warning: Unable to import deeppowers module, trying to rebuild..."
    ./build.sh
    python3 -c "import deeppowers as dp; print(f'CUDA available: {dp.cuda_available()}')"
}

# Run C++ examples
echo "Running C++ examples..."
if [ -f "../examples/basic_generation.cpp" ]; then
    if [ ! -f "../build/bin/basic_generation" ]; then
        echo "Example program not compiled, running build script..."
        ./build.sh
    fi
    ../build/bin/basic_generation
else
    echo "C++ example file does not exist"
fi

# Run Python API examples
echo "Running Python API examples..."
if [ -f "../examples/python_example.py" ]; then
    python3 ../examples/python_example.py
else
    echo "Python example file does not exist"
fi 