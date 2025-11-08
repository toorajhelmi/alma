"""Together.ai LLM wrapper."""
from .llm_base import Llm
from together import Together


class TogetherAI(Llm):
    """Wrapper for Together.ai models."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        """
        Initialize Together.ai wrapper.
        
        Args:
            api_key: Together.ai API key
            model: Model name (default: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
        """
        self.client = Together(api_key=api_key)
        self.model = model
        # Create a dummy instance for compatibility
        super().__init__(None)
    
    def send(self, prompt: str) -> str:
        """Send prompt to Together.ai and return response content."""
        # Estimate token count (rough estimate: 1 token ≈ 4 chars)
        estimated_input_tokens = len(prompt) // 4
        
        # For 405B model, Together.ai has strict limits: inputs + max_tokens <= 2048
        # Set max_tokens dynamically based on input size
        if "405B" in self.model:
            # Reserve space for input, set max_tokens to ensure total <= 2048
            max_tokens = max(100, 2048 - estimated_input_tokens - 50)  # 50 token buffer
        else:
            max_tokens = 2048  # Default for other models
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def get_token_count_fn(self):
        """Get a token counting function for Together.ai."""
        # Together.ai uses similar tokenization to GPT, approximate with tiktoken
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(encoding.encode(text))
        except ImportError:
            # Fallback to simple estimation
            return lambda text: max(1, len(text or "") // 4)

