"""Utilities for creating LLM instances (local or API)."""
from typing import Optional

try:
    from langchain.chat_models.ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

def create_local_llm(model_name: str = "llama3.1", base_url: Optional[str] = None) -> Optional:
    """
    Create a local LLM using Ollama.
    
    Args:
        model_name: Name of the Ollama model (default: llama3.1)
        base_url: Base URL for Ollama API (default: http://localhost:11434)
    
    Returns:
        ChatOllama instance or None if Ollama is not available
    """
    import os
    # Force CPU mode to avoid Metal hangs
    os.environ['OLLAMA_NO_METAL'] = '1'
    
    if ChatOllama is None:
        return None
    
    try:
        if base_url is None:
            base_url = "http://localhost:11434"
        
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
        )
    except Exception:
        return None


def check_ollama_available() -> bool:
    """Check if Ollama is available by attempting to create a local LLM."""
    llm = create_local_llm()
    return llm is not None

