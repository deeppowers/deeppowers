"""Model class for text generation."""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Callable
import time
import os
import numpy as np

try:
    import _deeppowers_core
except ImportError:
    print("Warning: _deeppowers_core not found, using mock implementation")
    class MockCore:
        def __init__(self):
            pass
        def load_model(self, *args, **kwargs):
            return self
        def get_config(self, *args, **kwargs):
            return {"model_type": "mock", "vocab_size": 50257}
        def forward(self, *args, **kwargs):
            return None
        def forward_batch(self, *args, **kwargs):
            return []
        def save(self, *args, **kwargs):
            pass
        def config(self, *args, **kwargs):
            return {}
        def device(self, *args, **kwargs):
            return "cpu"
        def to(self, *args, **kwargs):
            pass
        def model_type(self, *args, **kwargs):
            return "mock"
        def precision(self, *args, **kwargs):
            return 0
        def set_precision(self, *args, **kwargs):
            pass
        
        # Mock InferenceEngine
        class MockInferenceEngine:
            def __init__(self, *args, **kwargs):
                pass
            def generate(self, *args, **kwargs):
                class Result:
                    def __init__(self):
                        self.token_ids = [[1, 2, 3]]
                        self.logprobs = [[0.0, 0.0, 0.0]]
                        self.stop_reasons = ["mock"]
                        self.generation_time = 0.1
                return Result()
            def generate_batch(self, *args, **kwargs):
                return [self.generate()]
            def generate_stream(self, *args, **kwargs):
                callback = args[1]
                result = self.generate()
                callback(result)
            
        def InferenceEngine(self, *args, **kwargs):
            return self.MockInferenceEngine()
        
        class InferenceConfig:
            def __init__(self):
                self.max_length = 100
                self.min_length = 0
                self.temperature = 1.0
                self.top_k = 50
                self.top_p = 1.0
                self.repetition_penalty = 1.0
                self.num_return_sequences = 1
                self.do_sample = True
                self.early_stopping = False
                self.device = "cpu"
                
    _deeppowers_core = MockCore()

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_length: int = 100
    min_length: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = False

@dataclass
class GenerationResult:
    """Result of text generation."""
    
    texts: List[str]                       # Generated texts
    logprobs: Optional[List[float]] = None  # Token log probabilities
    tokens: Optional[List[List[str]]] = None  # Generated tokens
    stop_reasons: Optional[List[str]] = None  # Reasons for stopping
    generation_time: float = 0.0           # Generation time in seconds

class Model:
    """Model class for text generation."""
    
    def __init__(self):
        """Initialize the model."""
        self._model = None
        self._config = {}
        self._device = "cpu"
        self._model_type = None
        self._vocab_size = 0
        self._max_sequence_length = 2048
        self._tokenizer = None
        self._inference_engine = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs
    ) -> "Model":
        """Load a pre-trained model."""
        model = cls()
        model._device = device
        
        # Convert dtype string to DataType enum
        dtype_map = {
            "float32": _deeppowers_core.DataType.FLOAT32,
            "float16": _deeppowers_core.DataType.FLOAT16,
            "int8": _deeppowers_core.DataType.INT8,
            "int4": _deeppowers_core.DataType.INT4
        }
        
        # Convert device string to format expected by C++ backend
        device_cpp = device
        
        try:
            # Load model using C++ backend
            model._model = _deeppowers_core.load_model(model_name)
            
            # Move model to specified device
            if model._model.device() != device_cpp:
                model._model.to(device_cpp)
            
            # Get model configuration
            model._config = model._model.config()
            model._model_type = model._model.model_type()
            
            # Extract key properties
            model._vocab_size = int(model._config.get("vocab_size", 0))
            model._max_sequence_length = int(model._config.get("max_sequence_length", 2048))
            
            # Create inference engine
            model._inference_engine = _deeppowers_core.InferenceEngine(model._model)
            
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            print("Using mock model implementation")
            model._model = _deeppowers_core
            model._config = {"model_type": "mock", "vocab_size": 50257}
            model._model_type = "mock"
            model._vocab_size = 50257
            model._inference_engine = _deeppowers_core.InferenceEngine(model._model)
        
        return model
    
    def generate(
        self,
        input_ids: Union[List[int], List[List[int]]],
        attention_mask: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[List[int]]:
        """Generate text tokens."""
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        # Convert input to correct format if needed
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
            single_input = True
        else:
            single_input = False
        
        # Create default attention mask if none provided
        if attention_mask is None:
            attention_mask = [[1] * len(ids) for ids in input_ids]
        
        # Convert GenerationConfig to InferenceConfig
        inference_config = _deeppowers_core.InferenceConfig()
        inference_config.max_length = generation_config.max_length
        inference_config.min_length = generation_config.min_length
        inference_config.temperature = generation_config.temperature
        inference_config.top_k = generation_config.top_k
        inference_config.top_p = generation_config.top_p
        inference_config.repetition_penalty = generation_config.repetition_penalty
        inference_config.num_return_sequences = generation_config.num_return_sequences
        inference_config.do_sample = generation_config.do_sample
        inference_config.early_stopping = generation_config.early_stopping
        inference_config.device = self._device
        
        # Generate outputs
        if len(input_ids) == 1:
            # Single input case
            result = self._inference_engine.generate(input_ids[0], attention_mask[0], inference_config)
            output_ids = result.token_ids
        else:
            # Batch input case
            results = self._inference_engine.generate_batch(input_ids, attention_mask, inference_config)
            output_ids = []
            for result in results:
                output_ids.extend(result.token_ids)
        
        # Return in the same format as input
        if single_input and len(output_ids) == 1:
            return output_ids[0]
        return output_ids
    
    def save(self, path: str, format: str = "auto"):
        """Save model to file."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        
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
            self._model.save(path, format_cpp)
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @property
    def device(self) -> str:
        """Get current device."""
        if self._model is None:
            return self._device
        return self._model.device()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        if self._model is None:
            return self._config
        return self._model.config()
    
    def generate_stream(
        self,
        prompt: str,
        callback: Callable[[GenerationResult], bool],
        config: Optional[GenerationConfig] = None
    ) -> None:
        """Generate text with streaming output.
        
        Args:
            prompt: Input text prompt.
            callback: Function called for each generated chunk.
                     Should return True to continue generation.
            config: Generation configuration.
        """
        if config is None:
            config = GenerationConfig()
        
        # Tokenize input if tokenizer is available
        if self._tokenizer is not None:
            input_ids = self._tokenizer.encode(prompt)
        else:
            # For demo purposes, we'll use a simple dummy tokenization
            input_ids = [ord(c) % 1000 for c in prompt]
        
        # Convert GenerationConfig to InferenceConfig
        inference_config = _deeppowers_core.InferenceConfig()
        inference_config.max_length = config.max_length
        inference_config.min_length = config.min_length
        inference_config.temperature = config.temperature
        inference_config.top_k = config.top_k
        inference_config.top_p = config.top_p
        inference_config.repetition_penalty = config.repetition_penalty
        inference_config.num_return_sequences = config.num_return_sequences
        inference_config.do_sample = config.do_sample
        inference_config.early_stopping = config.early_stopping
        inference_config.device = self._device
        
        # Create streaming callback wrapper
        def stream_callback(cpp_result):
            # Convert C++ result to Python result
            result = GenerationResult(
                texts=[""] * len(cpp_result.token_ids),
                logprobs=[sum(lp) for lp in cpp_result.logprobs],
                tokens=[self._decode_tokens(ids) for ids in cpp_result.token_ids],
                stop_reasons=cpp_result.stop_reasons,
                generation_time=cpp_result.generation_time
            )
            
            # If tokenizer is available, decode the tokens
            if self._tokenizer is not None:
                for i, ids in enumerate(cpp_result.token_ids):
                    result.texts[i] = self._tokenizer.decode(ids)
            else:
                # Simple dummy decoding
                for i, ids in enumerate(cpp_result.token_ids):
                    result.texts[i] = "".join(chr(id % 128) for id in ids)
            
            # Call user callback
            return callback(result)
        
        # Run generation with streaming
        self._inference_engine.generate_stream(input_ids, stream_callback, inference_config)
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input text prompts.
            config: Generation configuration.
            
        Returns:
            List of GenerationResult, one for each prompt.
        """
        if config is None:
            config = GenerationConfig()
        
        # Tokenize inputs if tokenizer is available
        batch_input_ids = []
        if self._tokenizer is not None:
            batch_input_ids = self._tokenizer.encode_batch(prompts)
        else:
            # Simple dummy tokenization
            for prompt in prompts:
                batch_input_ids.append([ord(c) % 1000 for c in prompt])
        
        # Convert GenerationConfig to InferenceConfig
        inference_config = _deeppowers_core.InferenceConfig()
        inference_config.max_length = config.max_length
        inference_config.min_length = config.min_length
        inference_config.temperature = config.temperature
        inference_config.top_k = config.top_k
        inference_config.top_p = config.top_p
        inference_config.repetition_penalty = config.repetition_penalty
        inference_config.num_return_sequences = config.num_return_sequences
        inference_config.do_sample = config.do_sample
        inference_config.early_stopping = config.early_stopping
        inference_config.device = self._device
        
        # Generate outputs
        cpp_results = self._inference_engine.generate_batch(batch_input_ids, [], inference_config)
        
        # Convert C++ results to Python results
        results = []
        for cpp_result in cpp_results:
            result = GenerationResult(
                texts=[""] * len(cpp_result.token_ids),
                logprobs=[sum(lp) for lp in cpp_result.logprobs],
                tokens=[self._decode_tokens(ids) for ids in cpp_result.token_ids],
                stop_reasons=cpp_result.stop_reasons,
                generation_time=cpp_result.generation_time
            )
            
            # If tokenizer is available, decode the tokens
            if self._tokenizer is not None:
                for i, ids in enumerate(cpp_result.token_ids):
                    result.texts[i] = self._tokenizer.decode(ids)
            else:
                # Simple dummy decoding
                for i, ids in enumerate(cpp_result.token_ids):
                    result.texts[i] = "".join(chr(id % 128) for id in ids)
            
            results.append(result)
        
        return results
    
    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings."""
        # If we have a tokenizer, use it
        if self._tokenizer is not None:
            return self._tokenizer.convert_ids_to_tokens(token_ids)
        
        # Simple fallback
        return [f"token_{id}" for id in token_ids]
    
    @property
    def model_type(self) -> str:
        """Get the model type."""
        if self._model is None:
            return self._model_type
        return self._model.model_type()
    
    @property
    def model_path(self) -> str:
        """Get the model path."""
        return ""
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length."""
        return self._max_sequence_length
    
    def to_device(self, device: str) -> None:
        """Move the model to a device.
        
        Args:
            device: Device name ('cpu' or 'cuda').
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
            
        if device == self._device:
            return
            
        if self._model is not None:
            self._model.to(device)
            
        self._device = device
        
        # Reset inference engine to make sure it's using the new device
        if self._model is not None and self._inference_engine is not None:
            self._inference_engine = _deeppowers_core.InferenceEngine(self._model)
    
    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for this model.
        
        Args:
            tokenizer: Tokenizer instance.
        """
        self._tokenizer = tokenizer
    
    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        if self._model is None:
            self._config[key] = value
        else:
            # TODO: Implement set_config in C++ backend
            self._config[key] = value
    
    def get_config(self, key: str) -> str:
        """Get a configuration value.
        
        Args:
            key: Configuration key.
            
        Returns:
            Configuration value.
        """
        if self._model is None:
            return self._config.get(key, "")
        else:
            return self._config.get(key, "")
            
    def optimize(self, 
                optimization_type: str = "auto", 
                level: str = "o1", 
                enable_profiling: bool = False) -> Dict[str, Any]:
        """Optimize model for inference.
        
        Args:
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
        if self._model is None:
            raise RuntimeError("No model loaded")
            
        # Convert optimization type string to OptimizerType enum
        type_map = {
            "auto": _deeppowers_core.OptimizerType.AUTO,
            "fusion": _deeppowers_core.OptimizerType.FUSION,
            "pruning": _deeppowers_core.OptimizerType.PRUNING,
            "distillation": _deeppowers_core.OptimizerType.DISTILLATION,
            "quantization": _deeppowers_core.OptimizerType.QUANTIZATION,
            "caching": _deeppowers_core.OptimizerType.CACHING,
            "none": _deeppowers_core.OptimizerType.NONE
        }
        
        # Convert level string to OptimizationLevel enum
        level_map = {
            "none": _deeppowers_core.OptimizationLevel.NONE,
            "o1": _deeppowers_core.OptimizationLevel.O1,
            "o2": _deeppowers_core.OptimizationLevel.O2,
            "o3": _deeppowers_core.OptimizationLevel.O3
        }
        
        opt_type = type_map.get(optimization_type.lower(), _deeppowers_core.OptimizerType.AUTO)
        opt_level = level_map.get(level.lower(), _deeppowers_core.OptimizationLevel.O1)
        
        try:
            # Create optimizer config
            config = _deeppowers_core.OptimizerConfig()
            config.type = opt_type
            config.level = opt_level
            config.enable_profiling = enable_profiling
            
            # Create optimizer and apply optimizations
            optimizer = _deeppowers_core.InferenceOptimizer(config)
            result = optimizer.optimize(self._model)
            
            # Convert result to Python dictionary
            metrics = {
                "success": result.success,
                "speedup": result.speedup,
                "memory_reduction": result.memory_reduction,
                "accuracy_loss": result.accuracy_loss,
                "error_message": result.error_message
            }
            
            # Add detailed metrics
            for key, value in result.metrics.items():
                metrics[key] = value
                
            # Reset inference engine to use optimized model
            self._inference_engine = _deeppowers_core.InferenceEngine(self._model)
                
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
            
    def apply_quantization(self, precision: str = "int8") -> Dict[str, Any]:
        """Apply quantization to the model.
        
        Args:
            precision: Quantization precision. Options:
                - "int8": 8-bit integer quantization
                - "int4": 4-bit integer quantization
                - "mixed": Mixed precision quantization
                
        Returns:
            Dictionary with quantization results and metrics
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
            
        # Convert precision string to PrecisionMode enum
        precision_map = {
            "int8": _deeppowers_core.PrecisionMode.INT8,
            "int4": _deeppowers_core.PrecisionMode.INT4,
            "mixed": _deeppowers_core.PrecisionMode.MIXED,
            "full": _deeppowers_core.PrecisionMode.FULL,
            "auto": _deeppowers_core.PrecisionMode.AUTO
        }
        
        precision_mode = precision_map.get(precision.lower(), _deeppowers_core.PrecisionMode.INT8)
        
        try:
            # Create optimizer and apply quantization
            optimizer = _deeppowers_core.InferenceOptimizer(_deeppowers_core.OptimizerConfig())
            result = optimizer.apply_quantization(self._model, precision_mode)
            
            # Convert result to Python dictionary
            metrics = {
                "success": result.success,
                "speedup": result.speedup,
                "memory_reduction": result.memory_reduction,
                "accuracy_loss": result.accuracy_loss,
                "error_message": result.error_message
            }
            
            # Add detailed metrics
            for key, value in result.metrics.items():
                metrics[key] = value
                
            # Reset inference engine to use quantized model
            self._inference_engine = _deeppowers_core.InferenceEngine(self._model)
                
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Quantization failed: {e}")
            
    def benchmark(self, 
                 input_text: str = "Hello, world!", 
                 num_runs: int = 10,
                 warmup_runs: int = 3) -> Dict[str, float]:
        """Benchmark model inference performance.
        
        Args:
            input_text: Text to use for benchmarking
            num_runs: Number of inference runs to perform
            warmup_runs: Number of warmup runs before benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
            
        # Tokenize input if tokenizer is available
        if self._tokenizer is not None:
            input_ids = self._tokenizer.encode(input_text)
        else:
            # Simple dummy tokenization
            input_ids = [ord(c) % 1000 for c in input_text]
            
        # Create default attention mask
        attention_mask = [1] * len(input_ids)
        
        # Create default inference config
        inference_config = _deeppowers_core.InferenceConfig()
        inference_config.device = self._device
        
        # Perform warmup runs
        for _ in range(warmup_runs):
            self._inference_engine.generate(input_ids, attention_mask, inference_config)
            
        # Perform benchmark runs
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            result = self._inference_engine.generate(input_ids, attention_mask, inference_config)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate throughput (tokens per second)
        avg_tokens = sum(len(ids) for ids in result.token_ids) / len(result.token_ids)
        throughput = avg_tokens / (avg_latency / 1000)
        
        return {
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "throughput_tokens_per_sec": throughput,
            "num_runs": num_runs
        } 