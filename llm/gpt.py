"""GPT/OpenAI LLM wrapper."""
from .llm_base import Llm


class Gpt(Llm):
    """Wrapper for OpenAI's GPT models via LangChain ChatOpenAI."""
    
    def send(self, prompt: str) -> str:
        """Send prompt to GPT and return response content."""
        response = self.llm_instance.invoke(prompt)
        return getattr(response, "content", str(response))

