"""Model class for text generation."""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Callable
import time
import _deeppowers_core
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
        self._config = None
        self._device = "cuda"  # Default to CUDA if available
        self._model_type = None
        self._vocab_size = None
        self._max_sequence_length = None
    
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
        model._model = _deeppowers_core.load_model(
            model_name,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
        # Load model configuration
        model._config = model._model.get_config()
        model._model_type = model._config.get("model_type", "unknown")
        model._vocab_size = int(model._config.get("vocab_size", 0))
        model._max_sequence_length = int(model._config.get("max_sequence_length", 2048))
        
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
        
        # Create default attention mask if none provided
        if attention_mask is None:
            attention_mask = [[1] * len(ids) for ids in input_ids]
        
        # Call C++ backend for generation
        output_ids = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generation_config.max_length,
            min_length=generation_config.min_length,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            num_return_sequences=generation_config.num_return_sequences,
            do_sample=generation_config.do_sample,
            early_stopping=generation_config.early_stopping
        )
        
        return output_ids
    
    def save(self, path: str):
        """Save model to file."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        self._model.save(path)
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config
    
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
            
        start_time = time.time()
        
        # Convert prompt to token ids
        input_ids = self._model.tokenize(prompt)
        
        # Initialize generation state
        state = self._model.create_generation_state(
            input_ids,
            max_length=config.max_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty
        )
        
        # Generate tokens one by one
        while True:
            # Generate next token
            next_token, logprob = self._model.generate_next_token(state)
            
            # Check if generation should stop
            if next_token == self._model.eos_token_id or len(state.output_ids) >= config.max_length:
                break
                
            # Add token to state
            state.output_ids.append(next_token)
            state.logprobs.append(logprob)
            
            # Decode current output
            current_text = self._model.decode(state.output_ids)
            
            # Create result
            result = GenerationResult(
                texts=[current_text],
                logprobs=[sum(state.logprobs)],
                tokens=[self._model.convert_ids_to_tokens(state.output_ids)],
                generation_time=time.time() - start_time
            )
            
            # Call callback
            if not callback(result):
                break
    
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
            
        start_time = time.time()
        
        # Convert prompts to token ids
        batch_input_ids = [self._model.tokenize(prompt) for prompt in prompts]
        
        # Generate tokens
        batch_outputs = self._model.generate_batch(
            batch_input_ids,
            max_length=config.max_length,
            min_length=config.min_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            num_return_sequences=config.num_return_sequences,
            do_sample=config.do_sample,
            early_stopping=config.early_stopping
        )
        
        # Process results
        results = []
        for output in batch_outputs:
            result = GenerationResult(
                texts=[self._model.decode(ids) for ids in output["output_ids"]],
                logprobs=output.get("logprobs"),
                tokens=[self._model.convert_ids_to_tokens(ids) for ids in output["output_ids"]],
                stop_reasons=output.get("stop_reasons"),
                generation_time=time.time() - start_time
            )
            results.append(result)
            
        return results
    
    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._model_type
    
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
            
        self._model.to_device(device)
        self._device = device
    
    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
            
        self._model.set_config(key, value)
        self._config[key] = value
    
    def get_config(self, key: str) -> str:
        """Get a configuration value.
        
        Args:
            key: Configuration key.
            
        Returns:
            Configuration value.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")
            
        return self._model.get_config(key) 