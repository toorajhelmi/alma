"""LLM provider module for centralized LLM management."""
from .llm_base import Llm
from .llm_provider import LlmProvider
from .gpt import Gpt
from .llama import Llama
from .together import TogetherAI

# Module-level provider instance
_llm_provider = LlmProvider()

def get_provider() -> LlmProvider:
    """Get the module-level LLM provider instance."""
    return _llm_provider

# Convenience function matching the requested API
def GetLlm(llm: str = "", fallback: bool = True) -> Llm:
    """
    Get an LLM instance by name.
    
    Args:
        llm: Specific LLM name. If empty and fallback=True, returns default remote LLM.
        fallback: If requested LLM unavailable, fallback to default remote LLM
    
    Returns:
        Llm instance
    
    Raises:
        ValueError: If LLM not found and fallback=False
    """
    result = _llm_provider.get_llm(llm=llm, fallback=fallback)
    if result is None:
        raise ValueError(f"No LLM available (llm='{llm}')")
    return result

__all__ = ['Llm', 'LlmProvider', 'Gpt', 'Llama', 'TogetherAI', 'get_provider', 'GetLlm']

