import json
import os
import re
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from core.token_buffer_memory import TokenBufferMemory
from core.token_buffer_memory_with_summary import TokenBufferMemoryWithSummary
from core.token_buffer_memory_custom_summary import TokenBufferMemoryCustomSummary
from core.ofalma import apply_ofalma, apply_ofalma_rate_distortion
from core.experiment_utility import get_token_count_fn
from llm import GetLlm, get_provider, Gpt, Llama
from core.llm_utils import create_local_llm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INSTRUCTION = (
    'Return a strict JSON object with keys "answer" (integer) and optional '
    '"explanation" (string). If you return 0 for "answer", you MUST include '
    'an "explanation" describing why the answer is unknown or ambiguous. '
    'Do not add any other text.'
)

def get_user_messages(memory):
    """Extract user messages from memory."""
    # Handle custom_summary which doesn't have chat_memory
    if isinstance(memory, TokenBufferMemoryCustomSummary):
        # For custom summary, return empty list (it uses summary, not individual messages)
        return []
    
    source_messages = None
    if hasattr(memory, "chat_memory"):
        source_messages = memory.chat_memory.messages
    elif hasattr(memory, "memories"):
        candidates = getattr(memory, "memories", [])
        token_mem = next(
            (m for m in candidates 
             if isinstance(m, (ConversationTokenBufferMemory, TokenBufferMemory))), 
            None
        )
        if token_mem and hasattr(token_mem, "chat_memory"):
            source_messages = token_mem.chat_memory.messages
        else:
            any_mem = next(
                (m for m in candidates if hasattr(m, "chat_memory")), 
                None
            )
            if any_mem:
                source_messages = any_mem.chat_memory.messages
    
    if not source_messages:
        return []
    
    return [
        getattr(msg, "content", "")
        for msg in source_messages
        if getattr(msg, "type", type(msg).__name__).lower() in {"human", "user"}
        and getattr(msg, "content", "")
    ]


def get_history_text(memory):
    """Get the history text from memory (summary or messages)."""
    if hasattr(memory, "buffer") and memory.buffer:
        text = memory.buffer
    else:
        kept_user_texts = get_user_messages(memory)
        text = "\n".join(kept_user_texts) if kept_user_texts else ""
    
    if hasattr(memory, "buffer_as_str") and memory.buffer_as_str:
        text = memory.buffer_as_str
    
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(": "):
            line = line[2:]
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def parse_json_response(raw_content):
    """Parse JSON from LLM response, handling code fences and fallbacks."""
    candidate = raw_content.strip()
    
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start:end+1])
        except json.JSONDecodeError:
            pass
    
    m = re.search(r"-?\d+", candidate)
    num = int(m.group(0)) if m else 0
    return {"answer": num}


def create_conversational_chain(memory_type="token_buffer",
                                buffer_size=2048,
                                model_name="gpt-4o",
                                verbose=False,
                                memory_verbose=False):
    """Create a LangChain conversation with configurable memory type and size."""
    llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

    if memory_type == "token_buffer":
        memory = TokenBufferMemory(
            llm=llm,
            max_token_limit=buffer_size,
            memory_key="history",
            human_prefix="",
            ai_prefix="",
            verbose=memory_verbose,
        )
    elif memory_type == "summary":
        memory = TokenBufferMemoryWithSummary(
            llm=llm,
            return_messages=True,
            memory_key="history",
            max_token_limit=buffer_size,
            verbose=memory_verbose,
        )
    elif memory_type == "custom_summary":
        memory = TokenBufferMemoryCustomSummary(
            llm=llm,
            max_token_limit=buffer_size,
            verbose=memory_verbose,
        )
        # Note: custom_summary doesn't use ConversationChain, so return None
        # and handle differently in run_oneshot_from_memory
        return None, memory
    else:
        raise ValueError("Invalid memory_type")

    return ConversationChain(llm=llm, memory=memory, verbose=verbose)


def save_contexts_to_memory(memory, facts):
    """Save only facts to memory."""
    for fact in facts:
        memory.save_context({"input": fact}, {"output": ""})


def calculate_stats(chain, all_inputs, history_text, memory, final_prompt, facts):
    """Calculate token statistics."""
    # Handle custom_summary where chain might be None
    if chain is not None:
        token_fn = get_token_count_fn(chain.llm)
    else:
        # For custom_summary, use memory's token function
        token_fn = get_token_count_fn(memory.llm) if hasattr(memory, 'llm') else lambda x: max(1, len(x or "") // 4)
    
    total_tokens_facts = sum(token_fn(fact) for fact in facts)
    memory_limit = memory.max_token_limit if hasattr(memory, "max_token_limit") else None
    mem_token_size = token_fn(history_text)
    if isinstance(memory, TokenBufferMemoryCustomSummary):
        pruned_count = 0
    else:
        pruned_count = max(0, len(facts) - len(get_user_messages(memory)))
    ratio = (mem_token_size / total_tokens_facts) if total_tokens_facts else 0.0
    prompt_tokens_facts = token_fn(history_text)
    
    return {
        "total_tokens": total_tokens_facts,
        "memory_tokens": memory_limit,
        "final_prompt_tokens": prompt_tokens_facts,
        "facts_pruned": pruned_count,
        "memory_token_ratio": ratio,
    }


def run_oneshot_from_memory(chain, facts, question, verbose=False, evaluation_model: str = None, memory=None):
    """Feed facts into memory (with pruning), then send retained context + question + instruction."""
    all_inputs = facts + [question, INSTRUCTION]
    
    # Handle custom_summary which doesn't use ConversationChain
    if memory is not None:
        # Custom summary memory passed directly
        for fact in facts:
            memory.save_context({"input": fact}, {"output": ""})
        history_text = get_history_text(memory)
    else:
        # Standard LangChain memory
        memory = chain.memory
        save_contexts_to_memory(memory, facts)
        history_text = get_history_text(memory)
    
    final_prompt_parts = []
    if history_text:
        final_prompt_parts.append(history_text)
    if question:
        final_prompt_parts.append(question)
    if INSTRUCTION:
        final_prompt_parts.append(INSTRUCTION)
    final_prompt = "\n".join(final_prompt_parts)
    
    stats = calculate_stats(chain, all_inputs, history_text, memory, final_prompt, facts)
    
    if verbose:
        print(f"\n--- Sending to LLM ---")
        print(final_prompt)
        print(f"--- End of prompt ---\n")

    llm = GetLlm(llm=evaluation_model, fallback=True) if evaluation_model else GetLlm(fallback=True)
    raw_content = llm.send(final_prompt)
    parsed = parse_json_response(raw_content)

    return {
        "result": parsed,
        "stats": stats,
        "raw": raw_content,
    }


def print_results(outcome):
    """Print the results and statistics."""
    if not isinstance(outcome, dict) or "result" not in outcome:
        print("Output:", str(outcome).strip())
        return
    
    res = outcome.get("result", {})
    if isinstance(res, dict):
        if "answer" in res:
            print("Answer:", res.get("answer"))
        if res.get("explanation"):
            print("Explanation:", res.get("explanation"))
    
    stats = outcome.get("stats", {})
    print("Total tokens:", stats.get("total_tokens"))
    print("Memory tokens:", stats.get("memory_tokens"))
    print("Prompt tokens:", stats.get("final_prompt_tokens"))
    print("Facts Pruned:", stats.get("facts_pruned"))
    ratio = stats.get("memory_token_ratio")
    if ratio is not None:
        print("Memory/Total token ratio:", f"{ratio:.2f}")


FACTS = [
    "CRITICAL: There is a learning period of 10 days in the beginning - this is non-negotiable!",
    "Task A takes 2 days and is the foundation for all other work.",
    "Task C takes 3 days and shares the same machine as Task A.",
    "A and C share a machine — cannot overlap. This is a hard constraint.",
    "Task B takes 4 days and MUST start after both A and C finish.",
    "B must start after A finishes - no exceptions.",
    "B must start after C finishes - this dependency is critical.",
    "The team uses agile methodology for tracking progress.",
    "Daily standups help coordinate dependencies between tasks.",
    "Coffee is served in the breakroom every morning.",
    "The office building was painted blue last year.",
    "Team lunch happens on Fridays, usually around 12:30pm.",
    "The printer on the 3rd floor occasionally jams.",
    "There's free parking available in the visitor lot.",
    "Meals are catered for the project team during crunch time.",
    "Standup is at 9am sharp - attendance is mandatory.",
    "All code must be reviewed before merging to main branch.",
    "The office has ergonomic chairs in every cubicle.",
    "Tickets tracked in JIRA with custom workflow states.",
    "Repo is on Git and uses feature branch strategy.",
    "The water cooler runs out quickly on Mondays.",
    "WiFi password was changed last month to improve security."
]

QUESTION = ("Given these constraints, compute the minimum total project duration. "
            "Return only a single integer (the total days). If you cannot determine it, return 0.")

def run_ofalma(
    facts,
    question,
    buffer_size=160,
    verbose=False,
    impact_model: str = None,
    token_model: str = None,
    evaluation_model: str = None,
    display: bool = True,
):
    """Run OFALMA-based pruning and send to LLM."""
    if display:
        print("\n=== Memory type: ofalma ===")
    
    kept_facts, removed_facts, stats = apply_ofalma(facts, buffer_size, impact_model=impact_model, token_model=token_model)
    
    if verbose:
        print(f"\nOFALMA Pruning Stats:")
        print(f"  Total facts: {stats['total_facts']}")
        print(f"  Final kept: {stats['final_kept']}")
        print(f"  Facts removed: {len(removed_facts)}")
        
        if removed_facts:
            print("\n[Removed facts]")
            for idx, fact in enumerate(removed_facts, 1):
                fact_idx = stats['removed_indices'][idx - 1] if idx - 1 < len(stats['removed_indices']) else -1
                if fact_idx >= 0 and fact_idx < len(stats['impact_scores']):
                    score = stats['impact_scores'][fact_idx]
                    print(f"  {idx:>2} [{fact_idx+1}] (Imp:{score['importance']:.3f}) {fact}")
                else:
                    print(f"  {idx:>2} {fact}")
    
    final_prompt_parts = []
    if kept_facts:
        final_prompt_parts.append("\n".join(kept_facts))
    if question:
        final_prompt_parts.append(question)
    if INSTRUCTION:
        final_prompt_parts.append(INSTRUCTION)
    final_prompt = "\n".join(final_prompt_parts)
    
    llm = GetLlm(llm=evaluation_model, fallback=True) if evaluation_model else GetLlm(fallback=True)
    token_fn = llm.get_token_count_fn()
    total_tokens_facts = sum(token_fn(f) for f in facts)
    prompt_tokens_facts = sum(token_fn(f) for f in kept_facts)
    memory_tokens = sum(token_fn(f) for f in kept_facts)
    
    if verbose:
        print(f"\n--- Sending to LLM ---")
        print(final_prompt)
        print(f"--- End of prompt ---\n")
    
    raw_content = llm.send(final_prompt)
    parsed = parse_json_response(raw_content)
    
    outcome = {
        "result": parsed,
        "stats": {
            "total_tokens": total_tokens_facts,
            "memory_tokens": buffer_size,
            "final_prompt_tokens": prompt_tokens_facts,
            "facts_pruned": len(removed_facts),
            "memory_token_ratio": (memory_tokens / total_tokens_facts) if total_tokens_facts else 0.0,
        },
        "raw": raw_content,
    }
    
    if display:
        print_results(outcome)
    return outcome


def run_ofalma_rate_distortion(
    facts,
    question,
    buffer_size=160,
    verbose=False,
    k=3.0,
    condensation_model: str = None,
    impact_model: str = None,
    token_model: str = None,
    evaluation_model: str = None,
    display: bool = True,
):
    """Run OFALMA rate-distortion based condensation and send to LLM."""
    if display:
        print("\n=== Memory type: ofalma_rate_distortion ===")
    
    condensed_facts, stats = apply_ofalma_rate_distortion(
        facts, 
        buffer_size,
        k=k,
        condensation_model=condensation_model,
        impact_model=impact_model,
        token_model=token_model
    )
    
    if verbose:
        print(f"\nOFALMA Rate-Distortion Stats:")
        print(f"  Total facts: {stats['total_facts']}")
        print(f"  Condensed facts: {stats.get('condensed_facts_count', len(condensed_facts))}")
        print(f"  Removed facts: {stats.get('removed_facts_count', 0)}")
        print(f"  Alpha (scaling factor): {stats['alpha']:.3f}")
        print(f"  K (exponent): {stats['k']}")
        print(f"  Total tokens before: {stats['total_tokens_before']}")
        print(f"  Total tokens after: {stats['total_tokens_after']}")
        print(f"  Compression ratio: {stats['total_tokens_after'] / stats['total_tokens_before']:.3f}" if stats['total_tokens_before'] > 0 else "  N/A")
        
        # Print per-fact details
        if 'per_fact_stats' in stats:
            print(f"\nPer-Fact Condensation Details:")
            for pf in stats['per_fact_stats']:
                print(f"\n  Fact {pf['fact_idx'] + 1} (Importance: {pf['importance']:.3f}):")
                print(f"    Original ({pf['original_tokens']} tokens): {pf['original_fact'][:80]}{'...' if len(pf['original_fact']) > 80 else ''}")
                print(f"    Condensed ({pf['condensed_tokens']} tokens): {pf['condensed_fact'][:80]}{'...' if len(pf['condensed_fact']) > 80 else ''}")
                print(f"    Distortion ratio: {pf['distortion_ratio']:.3f} (target: {pf['target_tokens']}, ratio: {pf['final_ratio']:.3f})")
    
    final_prompt_parts = []
    if condensed_facts:
        final_prompt_parts.append("\n".join(condensed_facts))
    if question:
        final_prompt_parts.append(question)
    if INSTRUCTION:
        final_prompt_parts.append(INSTRUCTION)
    final_prompt = "\n".join(final_prompt_parts)
    
    llm = GetLlm(llm=evaluation_model, fallback=True) if evaluation_model else GetLlm(fallback=True)
    token_fn = llm.get_token_count_fn()
    total_tokens_facts = sum(token_fn(f) for f in facts)
    prompt_tokens_facts = sum(token_fn(cf) for cf in condensed_facts)
    memory_tokens = sum(token_fn(cf) for cf in condensed_facts)
    
    if verbose:
        print(f"\n--- Sending to LLM ---")
        print(final_prompt)
        print(f"--- End of prompt ---\n")
    
    raw_content = llm.send(final_prompt)
    parsed = parse_json_response(raw_content)
    
    outcome = {
        "result": parsed,
        "stats": {
            "total_tokens": total_tokens_facts,
            "memory_tokens": buffer_size,
            "final_prompt_tokens": prompt_tokens_facts,
            "facts_pruned": len(facts) - len(condensed_facts),  # All facts condensed, none removed
            "memory_token_ratio": (memory_tokens / total_tokens_facts) if total_tokens_facts else 0.0,
        },
        "raw": raw_content,
    }
    
    if display:
        print_results(outcome)
    return outcome


def run_replay_context(facts, question, verbose=False):
    """Run simulation with different memory types."""
    memory_types = ["token_buffer", "summary", "custom_summary"]
    for mtype in memory_types:
        print(f"\n=== Memory type: {mtype} ===")
        result = create_conversational_chain(
            memory_type=mtype,
            buffer_size=160,
            model_name="gpt-4o",
            verbose=False,
            memory_verbose=verbose
        )
        
        if mtype == "custom_summary":
            # Custom summary doesn't use ConversationChain
            chain, memory = result
            outcome = run_oneshot_from_memory(None, facts, question, verbose=verbose, memory=memory)
        else:
            chain = result
            outcome = run_oneshot_from_memory(chain, facts, question, verbose=verbose)
        
        print_results(outcome)




if __name__ == "__main__":
    # Register LLMs with provider
    provider = get_provider()
    
    # Register GPT-4o as remote default
    gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    provider.register('gpt4', Gpt(gpt_llm), default_remote=True)
    
    # Register Llama as local default (if available)
    # Commented out - not using Llama for now
    # llama_chat = create_local_llm(model_name="llama3.1")
    # if llama_chat is not None:
    #     provider.register('llama', Llama(llama_chat), default_local=True)
    #     print("Registered local Llama 3.1")
    # else:
    #     print("Warning: Local Llama not available. Will fallback to GPT-4o")
    
    VERBOSE = True
    
    # Test OFALMA (pruning approach)
    # run_ofalma(FACTS, QUESTION, buffer_size=160, verbose=VERBOSE)
    
    # Test OFALMA Rate-Distortion (condensation approach)
    # print("\n" + "="*60)
    # run_ofalma_rate_distortion(FACTS, QUESTION, buffer_size=160, verbose=VERBOSE, k=3.0)
    
    # Test Custom Summary
    print("\n" + "="*60)
    run_replay_context(FACTS, QUESTION, verbose=VERBOSE)
