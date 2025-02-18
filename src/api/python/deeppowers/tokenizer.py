import numpy as np
from typing import List, Union, Optional
from enum import Enum
import _deeppowers_core  # C++ bindings

class TokenizerType(Enum):
    BPE = "bpe"
    WORDPIECE = "wordpiece"

class Tokenizer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_type: TokenizerType = TokenizerType.BPE,
        vocab_file: Optional[str] = None
    ):
        """Initialize tokenizer with either a pre-trained model name or custom vocabulary."""
        self._tokenizer = _deeppowers_core.Tokenizer(tokenizer_type.value)
        
        if model_name:
            # Load pre-trained tokenizer configuration
            self._load_pretrained(model_name)
        elif vocab_file:
            # Load custom vocabulary
            self._tokenizer.initialize(vocab_file)
    
    def _load_pretrained(self, model_name: str):
        """Load pre-trained tokenizer configuration."""
        # TODO: Implement pre-trained model loading
        raise NotImplementedError("Pre-trained model loading not implemented yet")
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None
    ) -> Union[List[int], List[List[int]]]:
        """Encode text into token ids."""
        if isinstance(text, str):
            token_ids = self._tokenizer.encode(text)
            if max_length:
                token_ids = token_ids[:max_length]
            return token_ids
        else:
            batch_token_ids = self._tokenizer.encode_batch(text)
            if max_length:
                batch_token_ids = [ids[:max_length] for ids in batch_token_ids]
            if padding:
                max_len = max(len(ids) for ids in batch_token_ids)
                batch_token_ids = [
                    ids + [self._tokenizer.pad_token_id] * (max_len - len(ids))
                    for ids in batch_token_ids
                ]
            return batch_token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """Decode token ids back to text."""
        if isinstance(token_ids[0], int):
            return self._tokenizer.decode(token_ids)
        else:
            return self._tokenizer.decode_batch(token_ids)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        self._tokenizer.save(path)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        self._tokenizer.load(path)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token id."""
        return self._tokenizer.pad_token_id 