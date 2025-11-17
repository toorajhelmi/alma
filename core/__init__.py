"""Core FALMA modules."""
from .falma import apply_falma, apply_falma_rate_distortion, compute_impact_scores, llm_impact_factors
from .token_buffer_memory import TokenBufferMemory
from .token_buffer_memory_with_summary import TokenBufferMemoryWithSummary
from .experiment_utility import get_token_count_fn, prune_to_token_limit, truncate_text_to_token_limit

# Backward compatibility aliases
apply_ofalma = apply_falma
apply_ofalma_rate_distortion = apply_falma_rate_distortion

__all__ = [
    'apply_falma',
    'apply_falma_rate_distortion',
    'apply_ofalma',  # Backward compatibility
    'apply_ofalma_rate_distortion',  # Backward compatibility
    'compute_impact_scores',
    'llm_impact_factors',
    'TokenBufferMemory',
    'TokenBufferMemoryWithSummary',
    'get_token_count_fn',
    'prune_to_token_limit',
    'truncate_text_to_token_limit',
]

