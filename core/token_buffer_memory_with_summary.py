from langchain.memory import ConversationSummaryMemory
from .experiment_utility import get_token_count_fn, truncate_text_to_token_limit


class TokenBufferMemoryWithSummary(ConversationSummaryMemory):
    """ConversationSummaryMemory with token limit enforcement."""
    max_token_limit: int = 2048
    total_tokens: int = 0
    verbose: bool = False
    
    def __init__(self, *args, max_token_limit=None, verbose=False, **kwargs):
        object.__setattr__(self, "verbose", verbose)
        if max_token_limit is not None:
            kwargs["max_token_limit"] = max_token_limit
        super().__init__(*args, **kwargs)
    
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        
        count_tokens = get_token_count_fn(self.llm)
        
        summary = self.buffer if hasattr(self, "buffer") and self.buffer else ""
        if summary:
            max_limit = self.max_token_limit
            truncated, removed_text = truncate_text_to_token_limit(
                summary, count_tokens, max_limit
            )
            
            if removed_text:
                object.__setattr__(self, "buffer", truncated)
                self.total_tokens = self.total_tokens + count_tokens(truncated)
                if self.verbose:
                    print("[Memory summarized/pruned]")
                    print(f"{removed_text}")
            else:
                self.total_tokens = self.total_tokens + count_tokens(summary)

