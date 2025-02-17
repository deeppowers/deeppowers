# Deeppowers

<div align="center">
  <a href="deeppowers.xyz">
    <img src="https://github.com/deeppowers/deeppowers/blob/main/assets/deeppowers_logo.jpg" style="margin: 15px; max-width: 300px" width="30%" alt="Logo">
  </a>
</div>
<p align="center">
  <em>DeepPowers is a high-performance and flexible engine that accelerates the inference of large language models on heterogeneous GPUs. </em>
</p>


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/userguide.md)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/17)

## Overview

DeepPowers is a high-performance tokenizer implementation with memory optimization and parallel processing capabilities. It provides efficient text tokenization for large language models with features like WordPiece and BPE algorithms, memory pooling, and batch processing.

## Key Features

### Core Architecture
- Hardware Abstraction Layer (HAL)
- CUDA device management
- Basic tensor operations
- Kernel management system

### Request Processing
- Request queue management
- Batch processing system
- Priority scheduling
- Error handling mechanism

### Quantization
- INT8 quantization support
- INT4 quantization support
- Mixed-precision quantization
- Calibration data management

### API Interface
- C++ API infrastructure
- Python bindings
- REST API infrastructure
- gRPC service infrastructure

### Middleware
- Authentication middleware
- Rate limiting middleware
- Logging middleware
- Monitoring middleware
- Error handling middleware

## Architecture

### Technical Architecture Diagram

![Technical Architecture Diagram](assets/system-architecture.png)

The architecture follows a pipeline-based design with several key components:

1. **Request Flow**
   - User requests enter the system through a unified interface
   - Requests are queued and prioritized in the Request Queue
   - Batching system groups compatible requests for optimal processing
   - Execution Engine processes batches and generates results
   - Output is post-processed and returned to users

2. **Control Flow**
   - Configuration Manager oversees system settings and runtime parameters
   - Graph Compiler optimizes computation graphs for execution
   - Hardware Abstraction Layer provides unified access to different hardware backends

3. **Optimization Points**
   - Dynamic batching for throughput optimization
   - Graph compilation for computation optimization
   - Hardware-specific optimizations through HAL
   - Configuration-based performance tuning

### Directory Structure

```
deeppowers/
├── src/
│   ├── core/                      # Core implementation
│   │   ├── hal/                  # Hardware Abstraction Layer for device management
│   │   ├── request_queue/        # Request queue and management system
│   │   ├── batching/            # Batch processing and optimization
│   │   ├── execution/           # Execution engine and runtime
│   │   ├── distributed/         # Distributed computing support
│   │   ├── scheduling/          # Task scheduling and resource management
│   │   ├── monitoring/          # System monitoring and metrics
│   │   ├── config/             # Configuration management
│   │   ├── preprocessing/      # Input preprocessing pipeline
│   │   ├── postprocessing/     # Output postprocessing pipeline
│   │   ├── graph/              # Computation graph management
│   │   ├── api/               # Internal API implementations
│   │   ├── model/             # Base model architecture
│   │   ├── memory/            # Memory management system
│   │   ├── inference/         # Inference engine core
│   │   ├── models/            # Specific model implementations
│   │   ├── tokenizer/         # Tokenization implementations
│   │   └── utils/             # Utility components
│   ├── api/                   # External API implementations
│   └── common/                # Common utilities
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── examples/                  # Example usage  
├── docs/                      # Documentation
└── README.md                  # Project overview
```

The core module is organized into specialized components:

### Infrastructure Components
- **HAL (Hardware Abstraction Layer)**: Manages hardware devices and provides unified interface for different backends
- **Request Queue**: Handles incoming requests with priority management and load balancing
- **Batching**: Implements dynamic batching strategies for optimal throughput
- **Execution**: Core execution engine for model inference
- **Distributed**: Supports distributed computing and model parallelism

### Resource Management
- **Scheduling**: Manages task scheduling and resource allocation
- **Monitoring**: System metrics collection and performance monitoring
- **Config**: Configuration management and validation
- **Memory**: Advanced memory management and optimization

### Processing Pipeline
- **Preprocessing**: Input data preparation and normalization
- **Postprocessing**: Output processing and formatting
- **Graph**: Computation graph optimization and management
- **Inference**: Core inference engine implementation

### Model Components
- **Model**: Base model architecture and interfaces
- **Models**: Specific model implementations (GPT, BERT, etc.)
- **Tokenizer**: Text tokenization algorithms and utilities

### Support Systems
- **API**: Internal API implementations for core functionality
- **Utils**: Common utilities and helper functions

## Installation

### Prerequisites
- C++17 compiler
- CMake 3.15+
- Python 3.8+ (for Python bindings)
- ICU library for Unicode support

```bash
# Install dependencies (Ubuntu)
sudo apt-get install build-essential cmake libicu-dev

# Clone and build
git clone https://github.com/deeppowers/deeppowers.git
cd deeppowers
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# Install Python package (optional)
cd ./src/api/python
pip install -e .
```

```bash
# Clone the repository
git clone https://github.com/deeppowers/deeppowers.git
cd deeppowers

# Install dependencies
pip install -r requirements.txt

# Build from source
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Quick Start

```python
import deeppowers as dp

# Initialize tokenizer
tokenizer = dp.Tokenizer(tokenizer_type=dp.TokenizerType.WORDPIECE)

# Train on custom data
texts = ["your", "training", "texts"]
tokenizer.train(texts, vocab_size=30000, min_frequency=2)

# Save and load
tokenizer.save("tokenizer.model")
tokenizer.load("tokenizer.model")

# Basic usage
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Batch processing with parallel execution
texts = ["multiple", "texts", "for", "processing"]
tokens_batch = tokenizer.encode_batch_parallel(texts, batch_size=32)
```

## Performance Tuning

### Memory Optimization
```python
# Configure memory pool
tokenizer.set_memory_pool_size(4096)  # 4KB blocks
tokenizer.enable_string_pooling(True)

# Monitor memory usage
stats = tokenizer.get_memory_stats()
print(f"Memory pool usage: {stats['pool_usage']}MB")
print(f"String pool size: {stats['string_pool_size']}")
```

### Parallel Processing
```python
# Configure thread pool
tokenizer.set_num_threads(8)
tokenizer.set_batch_size(64)

# Process large datasets
with open("large_file.txt", "r") as f:
    texts = f.readlines()
tokens = tokenizer.encode_batch_parallel(texts)
```

## Documentation

- [User Guide](docs/userguide.md)
- [Architecture](docs/architecture.md)
- [Design Principles](docs/principles.md)

## Performance Optimization

DeepPowers includes several performance optimization features:
- Memory pooling and caching
- Dynamic batching
- Parallel processing
- Mixed-precision computation
- Distributed inference

## Roadmap

### In Progress
- Model execution framework
- Inference pipeline
- Streaming generation
- Dynamic batch processing
- Advanced memory management
- Computation graph optimization

### Planned
- Model implementations (GPT, BERT)
- Weight management system
- Model format conversion
- Custom model support
- Automatic mixed precision
- Dynamic shape support
- Operator fusion
- Auto-tuning system

## Benchmarking Tools

The project includes comprehensive benchmarking tools:

```bash
# Run performance benchmark
python scripts/benchmark.py --tokenizer model.bin --input test.txt

# Monitor memory usage
python scripts/memory_monitor.py --tokenizer model.bin --duration 60

# Process large datasets
python scripts/batch_process.py --input data/ --output results/
```


## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

Special thanks to all contributors and the open-source community.

## Contact

- GitHub Issues: [Create an issue](https://github.com/deeppowers/deeppowers/issues)
- Email: your.email@example.com
