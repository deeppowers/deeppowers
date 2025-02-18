from typing import List, Optional, Union, Dict
from .tokenizer import Tokenizer
from .model import Model, GenerationConfig

class Pipeline:
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
    ):
        """Initialize the pipeline with a model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs
    ) -> "Pipeline":
        """Create a pipeline from a pre-trained model."""
        model = Model.from_pretrained(model_name, device=device, dtype=dtype, **kwargs)
        tokenizer = Tokenizer(model_name=model_name)
        return cls(model=model, tokenizer=tokenizer)
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from input prompt(s)."""
        # Encode input text
        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True
        )
        
        # Create generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        # Generate tokens
        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
        
        # Decode output tokens
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Return single string if input was single string
        if isinstance(text, str) and num_return_sequences == 1:
            return outputs[0]
        return outputs
    
    def save(self, path: str):
        """Save both model and tokenizer."""
        self.model.save(f"{path}.model")
        self.tokenizer.save(f"{path}.tokenizer")
    
    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load both model and tokenizer."""
        model = Model()
        model.load(f"{path}.model")
        tokenizer = Tokenizer()
        tokenizer.load(f"{path}.tokenizer")
        return cls(model=model, tokenizer=tokenizer) 