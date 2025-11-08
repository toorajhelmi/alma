"""Base class for LLM wrappers."""
from abc import ABC, abstractmethod
from typing import Any


class Llm(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self, llm_instance: Any):
        """
        Initialize LLM wrapper.
        
        Args:
            llm_instance: The underlying LLM instance (e.g., ChatOpenAI, ChatOllama)
        """
        self.llm_instance = llm_instance
    
    @abstractmethod
    def send(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response content.
        
        Args:
            prompt: The prompt text to send
        
        Returns:
            The response content as a string
        """
        pass
    
    def get_token_count_fn(self):
        """Get a token counting function from the LLM instance."""
        token_fn = getattr(self.llm_instance, "get_num_tokens", None)
        if callable(token_fn):
            return token_fn
        return lambda text: max(1, len(text or "") // 4)

