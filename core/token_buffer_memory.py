from langchain.memory import ConversationTokenBufferMemory
from .experiment_utility import get_token_count_fn, prune_to_token_limit


class TokenBufferMemory(ConversationTokenBufferMemory):
    """Minimal pruning: remove oldest messages until new turn fits within max_token_limit."""
    total_tokens: int = 0
    verbose: bool = False
    
    def __init__(self, *args, verbose=False, **kwargs):
        object.__setattr__(self, "verbose", verbose)
        super().__init__(*args, **kwargs)
    
    def save_context(self, inputs, outputs):
        input_key = getattr(self, "input_key", None) or "input"
        output_key = getattr(self, "output_key", None) or "output"
        user_text = inputs.get(input_key, "") if isinstance(inputs, dict) else ""
        ai_text = outputs.get(output_key) if isinstance(outputs, dict) else None

        count_tokens = get_token_count_fn(self.llm)

        parts = []
        if user_text:
            parts.append(user_text)
        if ai_text and str(ai_text).strip():
            parts.append(str(ai_text))
        new_turn_text = "\n".join(parts)

        removed = []
        if new_turn_text:
            # Extract existing messages as strings
            existing_messages = [getattr(msg, "content", "") 
                                for msg in self.chat_memory.messages]
            
            # Check if new message fits with existing messages
            # We need to ensure the new message is always kept
            new_turn_tokens = count_tokens(new_turn_text)
            available_tokens = self.max_token_limit - new_turn_tokens
            
            # Prune existing messages to make room for the new one
            if available_tokens > 0:
                kept_existing, removed_existing = prune_to_token_limit(
                    existing_messages, count_tokens, available_tokens, reverse=False
                )
                
                # Map back to message objects for removal
                num_to_remove = len(existing_messages) - len(kept_existing)
                removed = [self.chat_memory.messages.pop(0) 
                          for _ in range(num_to_remove)]

        if user_text:
            self.chat_memory.add_user_message(user_text)
        if ai_text and str(ai_text).strip():
            self.chat_memory.add_ai_message(str(ai_text))
        
        if user_text:
            self.total_tokens = self.total_tokens + count_tokens(user_text)

        if removed and self.verbose:
            print("[Memory pruned]")
            for idx, msg in enumerate(removed, 1):
                content = getattr(msg, "content", "")
                print(f"  {idx:>2}:{content}")


