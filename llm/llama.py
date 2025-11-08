"""Ollama/Llama LLM wrapper."""
from .llm_base import Llm


class Llama(Llm):
    """Wrapper for Ollama/Llama models via LangChain ChatOllama."""
    
    def send(self, prompt: str) -> str:
        """Send prompt to Llama and return response content."""
        response = self.llm_instance.invoke(prompt)
        return getattr(response, "content", str(response))

