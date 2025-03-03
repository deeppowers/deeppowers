"""
DeepPowers - High Performance Text Generation Library
"""

from .tokenizer import Tokenizer, TokenizerType
from .model import Model, GenerationConfig, GenerationResult
from .pipeline import Pipeline
from .version import __version__
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Union

try:
    import _deeppowers_core
except ImportError:
    print("Warning: _deeppowers_core not found, using mock implementation")
    class MockCore:
        def __init__(self):
            pass
        def get_cuda_version(self):
            return "11.7"
        def is_cuda_available(self):
            return False
        def get_cuda_device_count(self):
            return 0
        def load_model(self, path, format=0):
            return None
        def convert_model(self, input_path, output_path, input_format, output_format):
            pass
        def optimize_model(self, model, optimization_type=0, level=1):
            return {"success": False, "error_message": "C++ backend not available"}
        class ModelFormat:
            AUTO = 0
            ONNX = 1
            PYTORCH = 2
            TENSORFLOW = 3
            CUSTOM = 4
        class OptimizerType:
            NONE = 0
            FUSION = 1
            PRUNING = 2
            DISTILLATION = 3
            QUANTIZATION = 4
            CACHING = 5
            AUTO = 6
        class OptimizationLevel:
            NONE = 0
            O1 = 1
            O2 = 2
            O3 = 3
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
    'optimize_model',
    'quantize_model',
    'benchmark_model',
    'get_model_config',
]

# Get models directory from environment variable or use default
MODELS_DIR = os.environ.get("DEEPPOWERS_MODELS_PATH", str(Path.home() / ".deeppowers" / "models"))

def _scan_models() -> Dict[str, Dict[str, Any]]:
    """Scan for available models in the models directory.
    
    Returns:
        Dictionary mapping model names to their configurations.
    """
    models = {}
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Scan for model directories
    for model_dir in Path(MODELS_DIR).iterdir():
        if not model_dir.is_dir():
            continue
            
        # Check for config.json
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue
            
        try:
            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Add to models dictionary
            models[model_dir.name] = {
                "path": str(model_dir),
                "config": config
            }
        except Exception as e:
            print(f"Error loading model config for {model_dir.name}: {e}")
            
    return models

# Scan for available models
AVAILABLE_MODELS = _scan_models()

def load_model(model_path: str, format: str = "auto", device: str = "cuda", dtype: str = "float16") -> Model:
    """Load a model from a file or directory.
    
    Args:
        model_path: Path to model file or directory.
        format: Model format ("auto", "onnx", "pytorch", "tensorflow").
        device: Device to load model on ("cpu", "cuda").
        dtype: Data type for model weights ("float32", "float16", "int8", "int4").
        
    Returns:
        Loaded model.
    """
    # Check if model_path is a known model name
    if model_path in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_path]["path"]
        
    # Convert format string to ModelFormat enum
    format_map = {
        "auto": _deeppowers_core.ModelFormat.AUTO,
        "onnx": _deeppowers_core.ModelFormat.ONNX,
        "pytorch": _deeppowers_core.ModelFormat.PYTORCH,
        "tensorflow": _deeppowers_core.ModelFormat.TENSORFLOW,
        "custom": _deeppowers_core.ModelFormat.CUSTOM
    }
    
    format_enum = format_map.get(format.lower(), _deeppowers_core.ModelFormat.AUTO)
    
    try:
        # Try to load using C++ backend
        model = Model.from_pretrained(model_path, device=device, dtype=dtype)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to Python implementation")
        
        # Fallback to Python implementation
        model = Model()
        model._model_path = model_path
        model._device = device
        
        return model

def convert_model(input_path: str, output_path: str, input_format: str, output_format: str) -> None:
    """Convert a model from one format to another.
    
    Args:
        input_path: Path to input model file or directory.
        output_path: Path to output model file or directory.
        input_format: Input model format ("auto", "onnx", "pytorch", "tensorflow").
        output_format: Output model format ("onnx", "pytorch", "tensorflow").
    """
    # Convert format strings to ModelFormat enums
    format_map = {
        "auto": _deeppowers_core.ModelFormat.AUTO,
        "onnx": _deeppowers_core.ModelFormat.ONNX,
        "pytorch": _deeppowers_core.ModelFormat.PYTORCH,
        "tensorflow": _deeppowers_core.ModelFormat.TENSORFLOW,
        "custom": _deeppowers_core.ModelFormat.CUSTOM
    }
    
    input_format_enum = format_map.get(input_format.lower(), _deeppowers_core.ModelFormat.AUTO)
    output_format_enum = format_map.get(output_format.lower(), _deeppowers_core.ModelFormat.ONNX)
    
    try:
        # Try to convert using C++ backend
        _deeppowers_core.convert_model(input_path, output_path, input_format_enum, output_format_enum)
    except Exception as e:
        raise RuntimeError(f"Error converting model: {e}")

def optimize_model(model: Model, 
                  optimization_type: str = "auto", 
                  level: str = "o1", 
                  enable_profiling: bool = False) -> Dict[str, Any]:
    """Optimize a model for inference.
    
    Args:
        model: Model to optimize.
        optimization_type: Type of optimization to apply. Options:
            - "auto": Automatically select optimizations
            - "fusion": Apply operator fusion
            - "pruning": Apply weight pruning
            - "quantization": Apply weight quantization
            - "caching": Apply KV-cache optimization
            - "none": No optimization
        level: Optimization aggressiveness level. Options:
            - "o1": Basic optimizations
            - "o2": Medium optimizations
            - "o3": Aggressive optimizations
        enable_profiling: Whether to collect performance metrics
        
    Returns:
        Dictionary with optimization results and metrics
    """
    return model.optimize(optimization_type, level, enable_profiling)

def quantize_model(model: Model, precision: str = "int8") -> Dict[str, Any]:
    """Apply quantization to a model.
    
    Args:
        model: Model to quantize.
        precision: Quantization precision. Options:
            - "int8": 8-bit integer quantization
            - "int4": 4-bit integer quantization
            - "mixed": Mixed precision quantization
            
    Returns:
        Dictionary with quantization results and metrics
    """
    return model.apply_quantization(precision)

def benchmark_model(model: Model, 
                   input_text: str = "Hello, world!", 
                   num_runs: int = 10,
                   warmup_runs: int = 3) -> Dict[str, float]:
    """Benchmark model inference performance.
    
    Args:
        model: Model to benchmark.
        input_text: Text to use for benchmarking
        num_runs: Number of inference runs to perform
        warmup_runs: Number of warmup runs before benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    return model.benchmark(input_text, num_runs, warmup_runs)

def list_available_models() -> List[str]:
    """List available pre-trained models.
    
    Returns:
        List of available model names.
    """
    return list(AVAILABLE_MODELS.keys())

def is_model_available(model_name: str) -> bool:
    """Check if a pre-trained model is available.
    
    Args:
        model_name: Name of the model to check.
        
    Returns:
        True if the model is available, False otherwise.
    """
    return model_name in AVAILABLE_MODELS

def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a pre-trained model.
    
    Args:
        model_name: Name of the model to get configuration for.
        
    Returns:
        Model configuration dictionary, or None if the model is not available.
    """
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]["config"]
    return None

def version() -> str:
    """Get the version of the library.
    
    Returns:
        Version string.
    """
    return "0.1.0"

def cuda_version() -> str:
    """Get the CUDA version.
    
    Returns:
        CUDA version string.
    """
    return _deeppowers_core.get_cuda_version()

def cuda_available() -> bool:
    """Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise.
    """
    return _deeppowers_core.is_cuda_available()

def cuda_device_count() -> int:
    """Get the number of CUDA devices.
    
    Returns:
        Number of CUDA devices.
    """
    return _deeppowers_core.get_cuda_device_count()

# Version information
__version__ = version() 