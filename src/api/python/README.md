# DeepPowers Python API

DeepPowers is a high-performance inference engine for large language models.

## Installation

```bash
pip install -e .
```

## Usage

```python
import deeppowers as dp

# Initialize model
model = dp.Model()

# Generate text
result = model.generate("Hello, world!")
print(result)
```

## Features

- High-performance inference
- Multiple model support
- Easy-to-use API
- GPU acceleration