#!/bin/bash

# Check if build is complete
if [ ! -d "../build" ]; then
    echo "Build directory does not exist, running build script..."
    ./build.sh
fi

# Check test data
if [ ! -f "../tests/data/sample.txt" ]; then
    echo "Test data does not exist, creating..."
    mkdir -p ../tests/data
    cat > ../tests/data/sample.txt << 'EOL'
This is a sample text for testing the tokenizer.
It contains multiple lines of text.
The tokenizer should be able to handle this correctly.
EOL
fi

# Create results directory
mkdir -p ../tests/results

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../build/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/../build/lib  # For macOS
export PYTHONPATH=$PYTHONPATH:$(pwd)/../src/api/python

# Run C++ unit tests
echo "Running C++ unit tests..."
cd ../build && ctest --output-on-failure
if [ $? -ne 0 ]; then
    echo "C++ unit tests failed"
    exit 1
fi

# Run Python tests
echo "Running Python tests..."
if [ -d "../tests/python" ]; then
    cd ../tests/python
    python3 -m pytest -v || {
        echo "Python tests failed"
        exit 1
    }
else
    echo "Python tests directory does not exist, skipping Python tests"
fi

# Run performance tests
echo "Running performance tests..."
if [ -f "benchmark.py" ]; then
    if [ -f "../build/lib/libtokenizer.so" ]; then
        python3 benchmark.py --tokenizer ../build/lib/libtokenizer.so \
                           --input ../tests/data/sample.txt \
                           --batch-size 32 \
                           --iterations 100
    else
        echo "Tokenizer library not found, skipping performance tests"
    fi
else
    echo "Performance test script not found, skipping performance tests"
fi

# Run memory monitoring tests
echo "Running memory monitoring tests..."
if [ -f "memory_monitor.py" ]; then
    if [ -f "../build/lib/libtokenizer.so" ]; then
        python3 memory_monitor.py --tokenizer ../build/lib/libtokenizer.so \
                                --input ../tests/data/sample.txt \
                                --output ../tests/results/memory.log \
                                --duration 60
    else
        echo "Tokenizer library not found, skipping memory monitoring tests"
    fi
else
    echo "Memory monitoring script not found, skipping memory monitoring tests"
fi

echo "Tests completed!" 