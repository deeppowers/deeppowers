"""
DeepPowers - High Performance Text Generation Library
"""

from .tokenizer import Tokenizer
from .model import Model, GenerationConfig, GenerationResult
from .pipeline import Pipeline
from .version import __version__

__all__ = [
    'Tokenizer',
    'Model',
    'Pipeline',
    'GenerationConfig',
    'GenerationResult',
    'load_model',
    'list_available_models',
    'is_model_available',
    'version',
    'cuda_version',
    'cuda_available',
    'cuda_device_count',
]

def load_model(model_path: str) -> Model:
    """Load a model from the specified path."""
    return Model(model_path)

def list_available_models() -> list[str]:
    """List all available models."""
    # TODO: Implement model discovery
    return []

def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    # TODO: Check model availability
    return False

def version() -> str:
    """Get the version of DeepPowers."""
    return __version__

def cuda_version() -> str:
    """Get the version of CUDA."""
    # TODO: Get from C++ API
    return "0.0.0"

def cuda_available() -> bool:
    """Check if CUDA is available."""
    # TODO: Get from C++ API
    return False

def cuda_device_count() -> int:
    """Get the number of CUDA devices."""
    # TODO: Get from C++ API
    return 0 