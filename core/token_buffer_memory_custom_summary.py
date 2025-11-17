"""Custom summarization memory that includes token limit in summarization prompt."""
from typing import List, Optional
from .experiment_utility import get_token_count_fn, truncate_text_to_token_limit
from llm import GetLlm


class TokenBufferMemoryCustomSummary:
    """Custom memory that summarizes facts with explicit token limit instructions."""
    
    def __init__(self, llm, max_token_limit: int = 160, verbose: bool = False):
        """
        Initialize custom summary memory.
        
        Args:
            llm: Language model instance (ChatOpenAI or compatible)
            max_token_limit: Maximum tokens allowed in summary
            verbose: Whether to print debug information
        """
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.verbose = verbose
        self.summary = ""  # Running summary of all facts
        self.total_tokens = 0
        self.token_count_fn = get_token_count_fn(llm)
        self.raw_facts: List[str] = []
        self.summary_generated = False
    
    def save_context(self, inputs, outputs):
        """
        Add a new fact and update the summary.
        
        Args:
            inputs: Dictionary with 'input' key containing the fact
            outputs: Dictionary with 'output' key (not used for facts)
        """
        new_fact = inputs.get("input", "")
        if not new_fact:
            return
        
        self.raw_facts.append(new_fact)
        self.summary_generated = False
    
    def _build_summarize_prompt(self, previous_summary: str, new_fact: str) -> str:
        """
        Build prompt for summarization that includes token limit instruction.
        
        Args:
            previous_summary: Existing summary (empty if first fact)
            new_fact: New fact to add
        
        Returns:
            Prompt string for LLM
        """
        max_tokens = self.max_token_limit
        
        if previous_summary:
            prompt = f"""You are maintaining a concise summary of conversation facts. 
A previous summary exists, and a new fact needs to be incorporated.

Previous summary:
{previous_summary}

New fact:
{new_fact}

Task: Create a NEW consolidated summary that incorporates the new fact into the previous summary. 
The summary must be AT MOST {max_tokens} tokens long. 
- Preserve all critical information from both the previous summary and the new fact
- Be concise but complete
- Focus on information relevant to answering analytical questions
- Do not exceed {max_tokens} tokens

Provide only the summary, no additional explanation:"""
        else:
            prompt = f"""Summarize the following fact concisely. 
The summary must be AT MOST {max_tokens} tokens long.

Fact:
{new_fact}

Task: Create a concise summary that preserves all critical information.
- Be concise but complete
- Focus on information relevant to answering analytical questions
- Do not exceed {max_tokens} tokens

Provide only the summary, no additional explanation:"""
        
        return prompt
    
    @property
    def buffer(self) -> str:
        """Get the current summary buffer."""
        self._ensure_summary()
        return self.summary
    
    @property
    def buffer_as_str(self) -> str:
        """Get the current summary buffer as string (alias for buffer)."""
        self._ensure_summary()
        return self.summary

    def _ensure_summary(self):
        """Generate a single summary covering all raw facts if needed."""
        if self.summary_generated:
            return

        if not self.raw_facts:
            self.summary = ""
            self.total_tokens = 0
            self.summary_generated = True
            return

        full_text = "\n".join(self.raw_facts)
        full_tokens = self.token_count_fn(full_text)

        if full_tokens <= self.max_token_limit:
            # Facts already fit; keep full text so higher budgets retain detail.
            self.summary = full_text
            self.total_tokens = full_tokens
            self.summary_generated = True
            return

        prompt = self._build_full_summary_prompt(full_text)

        try:
            summary_text = self._call_llm(prompt)
        except Exception as exc:
            if self.verbose:
                print(f"Error during summarization: {exc}")
            summary_text = full_text

        summary_tokens = self.token_count_fn(summary_text)
        if summary_tokens > self.max_token_limit:
            if self.verbose:
                print(f"[Summary exceeded limit] {summary_tokens} > {self.max_token_limit}, truncating...")
            truncated, removed_text = truncate_text_to_token_limit(
                summary_text, self.token_count_fn, self.max_token_limit
            )
            summary_text = truncated
            if self.verbose and removed_text:
                print(f"[Memory pruned] Removed from beginning: {removed_text[:100]}...")

        self.summary = summary_text
        self.total_tokens = self.token_count_fn(self.summary)
        self.summary_generated = True

    def _build_full_summary_prompt(self, combined_text: str) -> str:
        """Create prompt to summarize the entire dialogue at once."""
        max_tokens = self.max_token_limit
        return f"""You are compressing dialogue facts into a single concise summary.

Facts:
{combined_text}

Task: Produce ONE summary that preserves all critical information required to answer analytical questions.
- The summary must be AT MOST {max_tokens} tokens long.
- Prioritize constraints, durations, dependencies, and critical instructions.
- Omit trivia or low-impact details.
- Provide only the summary, no explanation.
"""

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM using provider wrapper with fallback."""
        llm_instance = None
        try:
            llm_instance = GetLlm(fallback=True)
        except Exception as exc:
            if self.verbose:
                print(f"[Warning] Unable to acquire provider LLM: {exc}")

        if llm_instance:
            return llm_instance.send(prompt)

        from langchain_core.messages import HumanMessage  # type: ignore[import]

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

