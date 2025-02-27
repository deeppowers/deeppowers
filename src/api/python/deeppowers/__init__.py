"""
DeepPowers - High Performance Text Generation Library
"""

from .tokenizer import Tokenizer
from .model import Model, GenerationConfig, GenerationResult
from .pipeline import Pipeline
from .version import __version__
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

try:
    import _deeppowers_core
except ImportError:
    print("Warning: _deeppowers_core not found, using mock implementation")
    class MockCore:
        def __init__(self):
            pass
        def get_cuda_version(self):
            return "0.0.0"
        def is_cuda_available(self):
            return False
        def get_cuda_device_count(self):
            return 0
        def load_model(self, path, format=0):
            return None
        def convert_model(self, input_path, output_path, input_format, output_format):
            pass
    _deeppowers_core = MockCore()

__all__ = [
    'Tokenizer',
    'Model',
    'Pipeline',
    'GenerationConfig',
    'GenerationResult',
    'load_model',
    'list_available_models',
    'is_model_available',
    'convert_model',
    'version',
    'cuda_version',
    'cuda_available',
    'cuda_device_count',
]

def _scan_models() -> Dict[str, Dict[str, Any]]:
    """Scan for available models in the models directory.
    
    The function looks for models in the directory specified by DEEPPOWERS_MODELS_PATH
    environment variable, or falls back to 'models' directory in current path.
    
    Expected directory structure:
    ```
    models/
    ├── model1/
    │   ├── config.json      # Model configuration
    │   ├── model.bin        # Model weights
    │   └── tokenizer.json   # Tokenizer configuration
    └── model2/
        ├── config.json
        ├── model.bin
        └── tokenizer.json
    ```
    
    Each model directory must contain a config.json file with at least:
    ```json
    {
        "model_type": "deepseek-v3",              # Model architecture type
        "description": "Model description", # Optional description
        "vocab_size": 67109856,               # Vocabulary size 671B
        "max_position_embeddings": 1024     # Maximum sequence length
    }
    ```
    """
    models = {}
    
    # Check environment variable for models path
    models_path = os.getenv('DEEPPOWERS_MODELS_PATH', 'models')
    if not os.path.exists(models_path):
        return models
        
    # Scan all subdirectories in models path
    for model_dir in Path(models_path).iterdir():
        if not model_dir.is_dir():
            continue
            
        # Check for config.json
        config_file = model_dir / 'config.json'
        if not config_file.exists():
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            model_name = model_dir.name
            models[model_name] = {
                'path': str(model_dir),
                'type': config.get('model_type', 'unknown'),
                'description': config.get('description', ''),
                'config': config
            }
        except Exception as e:
            print(f"Warning: Failed to load model config from {config_file}: {e}")
            
    return models

# Dynamic model discovery
_AVAILABLE_MODELS = _scan_models()

def load_model(model_path: str, format: str = "auto", device: str = "cuda", dtype: str = "float16") -> Model:
    """Load a model from the specified path.
    
    Args:
        model_path: Path to the model directory or model name
        format: Model format ('auto', 'onnx', 'pytorch', 'tensorflow', 'custom')
        device: Device to load the model on ('cpu' or 'cuda')
        dtype: Data type for model weights ('float32', 'float16', 'int8', 'int4')
    
    Returns:
        Model: Loaded model instance
    
    Examples:
        ```python
        # Load by model name
        model = load_model('your-model-name')
        
        # Load from specific path
        model = load_model('/path/to/my/model')
        
        # Load with specific format
        model = load_model('/path/to/my/model', format='onnx')
        
        # Load with custom configuration
        os.environ['DEEPPOWERS_MODELS_PATH'] = '/path/to/models'
        model = load_model('your-model-name', device='cpu', dtype='float32')
        ```
    """
    # Check if model_path is a model name in available models
    global _AVAILABLE_MODELS
    if model_path in _AVAILABLE_MODELS:
        model_path = _AVAILABLE_MODELS[model_path]['path']
    
    # Convert format string to ModelFormat enum
    format_map = {
        "auto": _deeppowers_core.ModelFormat.AUTO,
        "onnx": _deeppowers_core.ModelFormat.ONNX,
        "pytorch": _deeppowers_core.ModelFormat.PYTORCH,
        "tensorflow": _deeppowers_core.ModelFormat.TENSORFLOW,
        "custom": _deeppowers_core.ModelFormat.CUSTOM
    }
    format_cpp = format_map.get(format.lower(), _deeppowers_core.ModelFormat.AUTO)
    
    try:
        # Load model using C++ backend
        cpp_model = _deeppowers_core.load_model(model_path, format_cpp)
        
        # Create Python model wrapper
        model = Model()
        model._model = cpp_model
        model._device = device
        model._config = cpp_model.config()
        model._model_type = cpp_model.model_type()
        model._vocab_size = int(model._config.get("vocab_size", 0))
        model._max_sequence_length = int(model._config.get("max_sequence_length", 2048))
        
        # Move model to specified device if needed
        if cpp_model.device() != device:
            cpp_model.to(device)
        
        return model
    except Exception as e:
        print(f"Warning: Failed to load model using C++ backend: {e}")
        print("Falling back to Python implementation")
        return Model.from_pretrained(model_path, device=device, dtype=dtype)

def convert_model(input_path: str, output_path: str, input_format: str, output_format: str) -> None:
    """Convert a model between formats.
    
    Args:
        input_path: Path to input model
        output_path: Path to output model
        input_format: Input model format ('onnx', 'pytorch', 'tensorflow', 'custom')
        output_format: Output model format ('onnx', 'pytorch', 'tensorflow', 'custom')
    
    Examples:
        ```python
        # Convert PyTorch model to ONNX
        convert_model('model.pt', 'model.onnx', 'pytorch', 'onnx')
        
        # Convert TensorFlow model to PyTorch
        convert_model('model.pb', 'model.pt', 'tensorflow', 'pytorch')
        ```
    """
    # Convert format strings to ModelFormat enum
    format_map = {
        "auto": _deeppowers_core.ModelFormat.AUTO,
        "onnx": _deeppowers_core.ModelFormat.ONNX,
        "pytorch": _deeppowers_core.ModelFormat.PYTORCH,
        "tensorflow": _deeppowers_core.ModelFormat.TENSORFLOW,
        "custom": _deeppowers_core.ModelFormat.CUSTOM
    }
    
    input_format_cpp = format_map.get(input_format.lower(), _deeppowers_core.ModelFormat.AUTO)
    output_format_cpp = format_map.get(output_format.lower(), _deeppowers_core.ModelFormat.AUTO)
    
    try:
        _deeppowers_core.convert_model(input_path, output_path, input_format_cpp, output_format_cpp)
    except Exception as e:
        raise RuntimeError(f"Failed to convert model: {e}")

def list_available_models() -> List[str]:
    """List all available models.
    
    Returns:
        list[str]: List of available model names
    
    Examples:
        ```python
        # Set custom models directory
        os.environ['DEEPPOWERS_MODELS_PATH'] = '/path/to/models'
        
        # List all available models
        models = list_available_models()
        print(f"Available models: {models}")
        ```
    """
    global _AVAILABLE_MODELS
    _AVAILABLE_MODELS = _scan_models()
    return list(_AVAILABLE_MODELS.keys())

def is_model_available(model_name: str) -> bool:
    """Check if a model is available.
    
    Args:
        model_name: Name of the model to check
    
    Returns:
        bool: True if model is available, False otherwise
    
    Examples:
        ```python
        # Check if specific model is available
        if is_model_available('your-model-name'):
            model = load_model('your-model-name')
            
        # Check with custom models directory
        os.environ['DEEPPOWERS_MODELS_PATH'] = '/path/to/models'
        if is_model_available('your-model-name'):
            model = load_model('your-model-name')
        ```
    """
    global _AVAILABLE_MODELS
    _AVAILABLE_MODELS = _scan_models()
    return model_name in _AVAILABLE_MODELS

def version() -> str:
    """Get the version of DeepPowers."""
    return __version__

def cuda_version() -> str:
    """Get the version of CUDA."""
    return _deeppowers_core.get_cuda_version()

def cuda_available() -> bool:
    """Check if CUDA is available."""
    return _deeppowers_core.is_cuda_available()

def cuda_device_count() -> int:
    """Get the number of CUDA devices."""
    return _deeppowers_core.get_cuda_device_count() 