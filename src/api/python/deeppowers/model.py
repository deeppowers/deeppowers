"""Model class for text generation."""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import time
import _deeppowers_core

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
    
    def generate_stream(self, prompt: str,
                       callback: Callable[[GenerationResult], bool],
                       config: Optional[GenerationConfig] = None) -> None:
        """Generate text with streaming output.
        
        Args:
            prompt: Input text prompt.
            callback: Function called for each generated chunk.
                     Should return True to continue generation.
            config: Generation configuration.
        """
        if config is None:
            config = GenerationConfig(stream=True)
        else:
            config.stream = True
            
        start_time = time.time()
        # TODO: Call C++ model generate_stream
        # Placeholder streaming
        chunk = "Sample "
        for i in range(5):
            result = GenerationResult(
                texts=[chunk],
                generation_time=time.time() - start_time
            )
            if not callback(result):
                break
            chunk += "chunk "
            time.sleep(0.1)  # Simulate generation time
    
    def generate_batch(self, prompts: List[str],
                      config: Optional[GenerationConfig] = None) -> List[GenerationResult]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input text prompts.
            config: Generation configuration.
            
        Returns:
            List of GenerationResult, one for each prompt.
        """
        if config is None:
            config = GenerationConfig(batch_size=len(prompts))
        else:
            config.batch_size = len(prompts)
            
        start_time = time.time()
        # TODO: Call C++ model generate_batch
        # Placeholder batch generation
        results = []
        for prompt in prompts:
            generated_text = f"Sample batch text for: {prompt}"
            results.append(GenerationResult(
                texts=[generated_text],
                generation_time=time.time() - start_time
            ))
        
        return results
    
    @property
    def model_type(self) -> str:
        """Get the model type."""
        # TODO: Get from C++ model
        return "gpt"
    
    @property
    def model_path(self) -> str:
        """Get the model path."""
        return ""
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        # TODO: Get from C++ model
        return 50257
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length."""
        # TODO: Get from C++ model
        return 2048
    
    def to_device(self, device: str) -> None:
        """Move the model to a device.
        
        Args:
            device: Device name ('cpu' or 'cuda').
        """
        # TODO: Call C++ model to_device
        pass
    
    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        # TODO: Call C++ model set_config
        pass
    
    def get_config(self, key: str) -> str:
        """Get a configuration value.
        
        Args:
            key: Configuration key.
            
        Returns:
            Configuration value.
        """
        # TODO: Call C++ model get_config
        return "" 