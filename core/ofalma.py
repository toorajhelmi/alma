import math
import re
import json
from typing import List, Dict, Optional
from llm import GetLlm

theta = {
    "S": 0.5,
    "R": 0.2,
    "Q": 0.7,
    "E": 0.4
}

def llm_impact_factors_per_turn(dialogue, verbose=False, impact_model: str = None):
    """Ask LLM to compute impact factors one turn at a time (for weaker models)."""
    results = []
    
    for idx, utt in enumerate(dialogue):
        prompt = f"""You are analyzing a dialogue utterance to assign impact factors for memory compression.

Utterance: {utt}

Return a JSON object with:
- S: Surprisal (0–1, higher if this line introduces new or unexpected info)
- Q: RelevanceClarity (0–1, clarity and coherence with topic)
- E: Emphasis (0–1, strength, assertiveness, or emotional force)

Return ONLY a JSON object, no explanation.
Example format: {{"S": 0.3, "Q": 0.8, "E": 0.5}}"""
        
        try:
            if impact_model:
                llm = GetLlm(llm=impact_model, fallback=False)
            else:
                llm = GetLlm(fallback=True)  # Use default remote LLM
            content = llm.send(prompt)
        except Exception as e:
            if verbose:
                print(f"LLM call failed for turn {idx+1}: {e}")
            results.append({"S": 0.5, "Q": 0.5, "E": 0.5})
            continue
        
        candidate = content.strip()
        
        # Try to extract JSON from code fences first
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
        
        # Try to find JSON object pattern: {"S": ...}
        json_match = re.search(r'\{[\s\S]*?"S"[\s\S]*?\}', candidate)
        if json_match:
            candidate = json_match.group(0).strip()
        
        # Try parsing
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("Response is not a dict")
            # Ensure all keys exist
            result = {
                "S": data.get("S", 0.5),
                "Q": data.get("Q", 0.5),
                "E": data.get("E", 0.5)
            }
            results.append(result)
        except Exception:
            # If direct parse fails, try to find the object more aggressively
            try:
                start_idx = candidate.find('{')
                if start_idx != -1:
                    bracket_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(candidate)):
                        if candidate[i] == '{':
                            bracket_count += 1
                        elif candidate[i] == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        candidate = candidate[start_idx:end_idx]
                        data = json.loads(candidate)
                        result = {
                            "S": data.get("S", 0.5),
                            "Q": data.get("Q", 0.5),
                            "E": data.get("E", 0.5)
                        }
                        results.append(result)
                    else:
                        raise ValueError("No valid JSON found")
            except Exception as e:
                if verbose:
                    print(f"Error parsing JSON for turn {idx+1}: {candidate[:200]}")
                    print(f"Error: {e}")
                results.append({"S": 0.5, "Q": 0.5, "E": 0.5})
    
    return results


def llm_impact_factors(dialogue, verbose=False, per_turn=False, impact_model: str = None):
    """
    Ask LLM to compute impact factors for all utterances in the dialogue.
    
    Args:
        dialogue: List of utterance strings
        verbose: Whether to print debug information
        per_turn: If True, send each utterance separately (for weaker models).
                  If False, send entire dialogue at once (for stronger models).
    
    Returns:
        List of dicts with S, Q, E factors for each utterance
    """
    if per_turn:
        return llm_impact_factors_per_turn(dialogue, verbose=verbose, impact_model=impact_model)
    
    # Original batch approach for stronger models
    context = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(dialogue)])
    prompt = f"""
You are analyzing a dialogue to assign impact factors for memory compression.

Dialogue:
{context}

For each utterance, return a JSON array of objects, where each object has:
- S: Surprisal (0–1, higher if this line introduces new or unexpected info)
- Q: RelevanceClarity (0–1, clarity and coherence with topic)
- E: Emphasis (0–1, strength, assertiveness, or emotional force)

Return a JSON array with one object per utterance (same order as dialogue).
Example format: [{{"S": 0.3, "Q": 0.8, "E": 0.5}}, {{"S": 0.7, "Q": 0.6, "E": 0.4}}]
    """

    # Use specified model or default remote LLM
    try:
        if impact_model:
            llm = GetLlm(llm=impact_model, fallback=False)
        else:
            llm = GetLlm(fallback=True)  # Use default remote LLM
        content = llm.send(prompt)
    except Exception as e:
        if verbose:
            print(f"LLM call failed: {e}")
        content = ""
    
    candidate = content.strip()
    
    # Try to extract JSON from code fences first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()

    # Try to find JSON array pattern: [{"S": ...}, ...]
    json_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"S"[\s\S]*?\}[\s\S]*?\]', candidate)
    if json_match:
        candidate = json_match.group(0).strip()
    
    # Try parsing
    data = None
    try:
        data = json.loads(candidate)
        if not isinstance(data, list):
            raise ValueError("Response is not a list")
        if len(data) != len(dialogue):
            raise ValueError(f"Expected {len(dialogue)} items, got {len(data)}")
    except Exception:
        # If direct parse fails, try to find the array more aggressively
        try:
            # Find the first [ and last ] that contain valid JSON
            start_idx = candidate.find('[')
            if start_idx != -1:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(candidate)):
                    if candidate[i] == '[':
                        bracket_count += 1
                    elif candidate[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    candidate = candidate[start_idx:end_idx]
                    data = json.loads(candidate)
                    if not isinstance(data, list) or len(data) != len(dialogue):
                        raise ValueError("Invalid array")
        except Exception as e:
            if verbose:
                print(f"Error parsing JSON: {content[:500]}")
                print(f"Error: {e}")
            data = [
                {"S": 0.5, "Q": 0.5, "E": 0.5}
                for i in range(len(dialogue))
            ]

    # Ensure we always return a list
    if data is None:
        if verbose:
            print("Warning: Could not parse impact factors, using defaults")
        data = [
            {"S": 0.5, "Q": 0.5, "E": 0.5}
            for i in range(len(dialogue))
        ]

    return data

def compute_impact_scores(dialogue: List[str], verbose=False, impact_factors_list=None, impact_model: str = None) -> List[Dict]:
    total = len(dialogue)
    results = []

    if impact_factors_list is None:
        all_factors = llm_impact_factors(dialogue, verbose=verbose, impact_model=impact_model)
    else:
        all_factors = impact_factors_list

    # Handle case where llm_impact_factors returns None (error case)
    if all_factors is None:
        if verbose:
            print("Warning: llm_impact_factors returned None, using default values")
        all_factors = [{"S": 0.5, "Q": 0.5, "E": 0.5} for _ in dialogue]

    for idx, utt in enumerate(dialogue):
        data = all_factors[idx] if idx < len(all_factors) else {}
        
        S = data.get("S", 0.5)
        R = (idx + 1) / total
        Q = data.get("Q", 0.5)
        E = data.get("E", 0.5)
        
        z = (theta["S"] * S +
             theta["R"] * R +
             theta["Q"] * Q +
             theta["E"] * E)
        importance = 1 / (1 + math.exp(-z))

        results.append({
            "utterance": utt,
            "S": round(S, 3),
            "R": round(R, 3),
            "Q": round(Q, 3),
            "E": round(E, 3),
            "importance": round(importance, 3)
        })

    return results


def apply_ofalma(facts: List[str], max_tokens: int, impact_factors_list=None, impact_model: str = None, token_model: str = None):
    """Prune facts using OFALMA scores: sort by importance and drop least important until fit token limit."""
    if token_model:
        llm = GetLlm(llm=token_model, fallback=False)
    else:
        llm = GetLlm(fallback=True)  # Use default remote LLM for token counting
    impact_scores = compute_impact_scores(facts, verbose=False, impact_factors_list=impact_factors_list, impact_model=impact_model)
    token_fn = llm.get_token_count_fn()
    
    # Create list of (index, fact, importance) tuples and sort by importance (descending)
    fact_data = [(idx, facts[idx], score['importance']) 
                 for idx, score in enumerate(impact_scores)]
    fact_data.sort(key=lambda x: x[2], reverse=True)  # Sort by importance descending
    
    # Keep most important facts until token limit
    kept = []
    kept_indices = []
    current_tokens = 0
    
    for idx, fact, importance in fact_data:
        item_tokens = token_fn(fact)
        if current_tokens + item_tokens <= max_tokens:
            kept.append(fact)
            kept_indices.append(idx)
            current_tokens += item_tokens
        else:
            break
    
    # Sort kept facts by original index to maintain order
    kept_with_indices = list(zip(kept, kept_indices))
    kept_with_indices.sort(key=lambda x: x[1])
    final_kept = [fact for fact, idx in kept_with_indices]
    final_kept_indices = sorted(kept_indices)
    
    # Get removed facts and indices
    removed_indices = [idx for idx in range(len(facts)) if idx not in final_kept_indices]
    all_removed_facts = [facts[idx] for idx in removed_indices]
    
    stats = {
        'final_kept': len(final_kept),
        'total_facts': len(facts),
        'impact_scores': impact_scores,
        'final_kept_indices': final_kept_indices,
        'removed_indices': removed_indices
    }
    
    return final_kept, all_removed_facts, stats


def condense_fact(fact: str, target_tokens: int, token_fn=None, condensation_model: str = None) -> str:
    """Condense a fact to approximately target_tokens using specified model."""
    # Get condensation model from provider
    if condensation_model:
        llm = GetLlm(llm=condensation_model, fallback=False)
    else:
        llm = GetLlm(fallback=True)  # Use default remote LLM
    
    if token_fn is None:
        token_fn = llm.get_token_count_fn()
    
    prompt = f"""Summarize the following text in approximately {target_tokens} tokens, preserving all key information and important details:

{fact}

Provide a concise summary:"""
    
    try:
        condensed = llm.send(prompt)
        
        # If condensed version is still too long, truncate
        actual_tokens = token_fn(condensed)
        if actual_tokens > target_tokens * 1.2:  # Allow 20% over target
            # Truncate to target
            words = condensed.split()
            truncated = ""
            for word in words:
                test = truncated + (" " if truncated else "") + word
                if token_fn(test) <= target_tokens:
                    truncated = test
                else:
                    break
            condensed = truncated if truncated else condensed[:target_tokens * 4]  # Fallback
        
        return condensed.strip()
    except Exception as e:
        # Last resort: simple truncation
        words = fact.split()
        truncated = ""
        for word in words:
            test = truncated + (" " if truncated else "") + word
            if token_fn(test) <= target_tokens:
                truncated = test
            else:
                break
        return truncated if truncated else fact[:target_tokens * 4]


def apply_ofalma_rate_distortion(
    facts: List[str], 
    max_tokens: int, 
    impact_factors_list=None,
    k: float = 3.0,
    condensation_model: str = None,
    impact_model: str = None,
    token_model: str = None
):
    """
    Rate-distortion based pruning: condense facts based on importance.
    
    Formula: condensation_ratio = alpha * e^(k * importance) / e^k
    Where alpha is scaling factor to ensure sum(c_i) <= m
    This means higher importance → less condensation (more tokens preserved)
    
    Args:
        facts: List of fact strings
        max_tokens: Maximum total tokens allowed
        impact_factors_list: Pre-computed impact factors (optional)
        k: Exponential factor (default 3.0)
        condensation_model: Model name for condensation (optional, uses default if None)
        impact_model: Model name for computing impact factors (optional, uses default if None)
        token_model: Model name for token counting (optional, uses default if None)
    
    Returns:
        Tuple of (condensed_facts, stats_dict)
    """
    # Get models from provider
    if condensation_model:
        llm = GetLlm(llm=condensation_model, fallback=False)
    else:
        llm = GetLlm(fallback=True)  # Use default remote LLM
    
    if token_model:
        token_llm = GetLlm(llm=token_model, fallback=False)
    else:
        token_llm = llm  # Use same model as condensation for token counting
    
    impact_scores = compute_impact_scores(facts, verbose=False, 
                                         impact_factors_list=impact_factors_list,
                                         impact_model=impact_model)
    token_fn = token_llm.get_token_count_fn()
    
    # Calculate base condensation ratios using exponential formula: e^(k*importance) / e^k
    # This ensures: importance=0 → ratio=0, importance=1 → ratio=1
    e_k = math.exp(k)  # Normalization factor: e^k for normalization
    
    fact_data = []
    for idx, score in enumerate(impact_scores):
        importance = score['importance']
        base_ratio = math.exp(k * importance) / e_k
        original_tokens = token_fn(facts[idx])
        fact_data.append({
            'idx': idx,
            'fact': facts[idx],
            'importance': importance,
            'base_ratio': base_ratio,
            'original_tokens': original_tokens
        })
    
    # Calculate initial condensed sizes
    total_initial_tokens = sum(item['original_tokens'] * item['base_ratio'] 
                              for item in fact_data)
    
    # Apply scaling factor to use full memory budget
    # If under limit, scale up; if over limit, scale down
    alpha = 1.0
    if total_initial_tokens > max_tokens:
        alpha = max_tokens / total_initial_tokens  # Scale down to fit
    elif 0 < total_initial_tokens < max_tokens:
        alpha = max_tokens / total_initial_tokens  # Scale up to use full budget
    # If total_initial_tokens is 0 or exactly equals max_tokens, keep alpha = 1.0
    
    # Calculate target tokens for each fact
    for item in fact_data:
        item['final_ratio'] = item['base_ratio'] * alpha
        item['target_tokens'] = round(item['original_tokens'] * item['final_ratio'])
    
    # Condense each fact (skip if target_tokens = 0)
    condensed_facts = []
    for item in fact_data:
        if item['target_tokens'] == 0:
            condensed_facts.append("")  # Empty string for 0-token facts (will be filtered later)
        else:
            condensed = condense_fact(
                item['fact'], 
                item['target_tokens'], 
                token_fn=token_fn,
                condensation_model=condensation_model
            )
            condensed_facts.append(condensed)
    
    # Verify total tokens (condensed facts may be slightly longer than target)
    total_actual_tokens = sum(token_fn(cf) for cf in condensed_facts)
    
    # If still over limit, further condense lowest importance facts
    if total_actual_tokens > max_tokens:
        # Sort by importance (lowest first) and condense more
        fact_data_with_condensed = [(item, condensed) for item, condensed in zip(fact_data, condensed_facts)]
        fact_data_with_condensed.sort(key=lambda x: x[0]['importance'])
        
        excess = total_actual_tokens - max_tokens
        current_total = total_actual_tokens
        
        for i, (item, condensed) in enumerate(fact_data_with_condensed):
            if current_total <= max_tokens:
                break
            
            current_tokens = token_fn(condensed)
            if current_tokens > 0:
                # Reduce target by excess proportion
                reduction_ratio = min(0.9, excess / current_total)  # Cap at 90% reduction
                new_target = round(current_tokens * (1 - reduction_ratio))
                if new_target == 0:
                    condensed_new = ""  # Remove fact entirely
                else:
                    condensed_new = condense_fact(
                        item['fact'],
                        new_target,
                        token_fn=token_fn,
                        condensation_model=condensation_model
                    )
                fact_data_with_condensed[i] = (item, condensed_new)
                current_total = current_total - current_tokens + token_fn(condensed_new)
        
        # Rebuild in original order (by index)
        fact_data_with_condensed.sort(key=lambda x: x[0]['idx'])
        condensed_facts = [cf for _, cf in fact_data_with_condensed]
    
    # Filter out empty facts (0 tokens) before returning
    final_condensed_facts = []
    for cf in condensed_facts:
        if cf and token_fn(cf) > 0:  # Keep non-empty facts
            final_condensed_facts.append(cf)
    
    # Calculate per-fact stats - include all facts, even those removed (0 tokens)
    per_fact_stats = []
    for item in fact_data:
        idx = item['idx']
        condensed = condensed_facts[idx] if idx < len(condensed_facts) else ""
        original_tokens = item['original_tokens']
        condensed_tokens = token_fn(condensed) if condensed else 0
        distortion_ratio = condensed_tokens / original_tokens if original_tokens > 0 else 0.0
        per_fact_stats.append({
            'fact_idx': idx,
            'original_fact': item['fact'],
            'condensed_fact': condensed if condensed else "[REMOVED - 0 tokens]",
            'original_tokens': original_tokens,
            'condensed_tokens': condensed_tokens,
            'distortion_ratio': distortion_ratio,
            'importance': item['importance'],
            'target_tokens': item['target_tokens'],
            'final_ratio': item['final_ratio']
        })
    
    # Sort by original index to maintain order
    per_fact_stats.sort(key=lambda x: x['fact_idx'])
    
    stats = {
        'total_facts': len(facts),
        'condensed_facts_count': len(final_condensed_facts),
        'removed_facts_count': len(facts) - len(final_condensed_facts),
        'condensation_applied': True,
        'impact_scores': impact_scores,
        'alpha': alpha,
        'k': k,
        'total_tokens_before': sum(token_fn(f) for f in facts),
        'total_tokens_after': sum(token_fn(cf) for cf in final_condensed_facts),
        'per_fact_stats': per_fact_stats
    }
    
    return final_condensed_facts, stats
