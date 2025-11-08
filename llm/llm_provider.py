"""LLM provider for centralized LLM management."""
from typing import Dict, Optional
from .llm_base import Llm


class LlmProvider:
    """Provider for managing and accessing LLMs."""
    
    def __init__(self):
        """Initialize the provider."""
        self._llms: Dict[str, Llm] = {}
        self._local_default: Optional[str] = None
        self._remote_default: Optional[str] = None
    
    def register(self, name: str, llm: Llm, 
                 default_local: bool = False, default_remote: bool = False):
        """
        Register an LLM with the provider.
        
        Args:
            name: Unique name for the LLM (e.g., 'gpt4', 'llama')
            llm: Llm instance to register
            default_local: Set as default local LLM (deprecated, kept for compatibility)
            default_remote: Set as default remote LLM
        """
        self._llms[name] = llm
        
        if default_local:
            self._local_default = name
        if default_remote:
            self._remote_default = name
    
    def get_llm(self, llm: str = "", fallback: bool = True) -> Optional[Llm]:
        """
        Get an LLM instance by name.
        
        Args:
            llm: Specific LLM name to retrieve. If empty and fallback=True, returns default remote LLM.
            fallback: If requested LLM unavailable, fallback to default remote LLM
        
        Returns:
            Llm instance or None if not found and fallback=False
        """
        if llm:
            # Requested specific LLM
            if llm in self._llms:
                return self._llms[llm]
            elif fallback:
                # Fallback to default remote LLM
                return self._get_default_remote()
            else:
                raise ValueError(f"LLM '{llm}' not found and fallback is disabled")
        else:
            # Return default remote LLM
            return self._get_default_remote()
    
    def _get_default_remote(self) -> Optional[Llm]:
        """Get default remote LLM."""
        if self._remote_default and self._remote_default in self._llms:
            return self._llms[self._remote_default]
        return None
    
    def _get_default(self, local: bool) -> Optional[Llm]:
        """Get default LLM for local or remote (deprecated, use _get_default_remote)."""
        if local:
            if self._local_default and self._local_default in self._llms:
                return self._llms[self._local_default]
        else:
            return self._get_default_remote()
        return None
    
    def is_available(self, name: str) -> bool:
        """Check if an LLM is registered."""
        return name in self._llms

