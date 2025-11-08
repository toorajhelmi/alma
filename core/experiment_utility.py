"""Shared utilities for different memory/pruning approaches."""
from typing import List, Callable, Tuple


def get_token_count_fn(llm) -> Callable[[str], int]:
    """Get a token counting function from an LLM."""
    token_fn = getattr(llm, "get_num_tokens", None)
    if callable(token_fn):
        return token_fn
    return lambda text: max(1, len(text or "") // 4)


def prune_to_token_limit(items: List[str], token_fn: Callable[[str], int], 
                         max_tokens: int, reverse: bool = False) -> Tuple[List[str], List[str]]:
    """Prune items to fit within token limit."""
    if reverse:
        kept = []
        removed = []
        current_tokens = 0
        
        for item in reversed(items):
            item_tokens = token_fn(item)
            if current_tokens + item_tokens <= max_tokens:
                kept.insert(0, item)
                current_tokens += item_tokens
            else:
                removed.insert(0, item)
        
        return kept, removed
    else:
        kept = []
        removed = []
        current_tokens = 0
        
        for item in items:
            item_tokens = token_fn(item)
            if current_tokens + item_tokens <= max_tokens:
                kept.append(item)
                current_tokens += item_tokens
            else:
                removed.append(item)
        
        return kept, removed


def truncate_text_to_token_limit(text: str, token_fn: Callable[[str], int], 
                                  max_tokens: int) -> Tuple[str, str]:
    """Truncate text from the beginning to fit within token limit."""
    if token_fn(text) <= max_tokens:
        return text, ""
    
    words = text.split()
    truncated = text
    removed_words = []
    
    for i in range(len(words)):
        truncated = " ".join(words[i:])
        if token_fn(truncated) <= max_tokens:
            removed_words = words[:i]
            break
    
    removed_text = " ".join(removed_words) if removed_words else ""
    return truncated, removed_text


