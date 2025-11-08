"""Data generator for creating dialogues with known answers using GPT."""
import random
import json
from typing import List, Dict, Tuple
from llm import GetLlm

INSTRUCTION = (
    'Return a strict JSON object with keys "answer" (integer) and optional '
    '"explanation" (string). If you return 0 for "answer", you MUST include '
    'an "explanation" describing why the answer is unknown or ambiguous. '
    'Do not add any other text.'
)


class DataGenerator:
    """Generate dialogues with known answers using GPT."""
    
    def __init__(self, seed=42, model_name="gpt-4o"):
        random.seed(seed)
        self.model_name = model_name
        self.problem_types = [
            "project scheduling",
            "logic puzzle",
            "mathematical calculation",
            "resource allocation",
            "sequence pattern",
            "optimization problem",
            "constraint satisfaction",
        ]
    
    def generate(self, n: int = 1, validate: bool = True) -> List[Dict]:
        """Generate n dialogues with known answers using GPT."""
        dialogues = []
        failures = 0
        attempts = 0
        max_attempts = n * 20  # Safety limit
        
        i = 0
        while i < n and attempts < max_attempts:
            attempts += 1
            print(f"Generating dialogue {i+1}/{n} (attempt {attempts})...")
            problem_type = random.choice(self.problem_types)
            dialogue, answer, justification, problem_type_used = self._generate_with_gpt(problem_type)
            
            # Validate the answer by asking GPT to solve it
            if validate:
                gpt_answer = self._validate_dialogue(dialogue)
                if gpt_answer != answer:
                    failures += 1
                    failure_rate = failures / attempts
                    print(f"  ❌ Validation failed: Expected {answer}, got {gpt_answer}. Dropping this dialogue.")
                    
                    # Stop if failure rate exceeds 50% (more than 1 out of 2)
                    # Only check after at least 10 attempts to get meaningful statistics
                    if attempts >= 10 and failure_rate > 0.5:
                        print(f"\n⚠️  ERROR: Validation failure rate ({failure_rate*100:.1f}%) exceeds 50%!")
                        print(f"   Failures: {failures}/{attempts} attempts")
                        print(f"   Stopping generation to avoid generating incorrect cases.")
                        break
                    continue  # Skip this dialogue
            
            dialogue_data = {
                "id": i + 1,
                "dialogue": dialogue,
                "answer": answer,
                "justification": justification,
                "problem_type": problem_type_used
            }
            
            dialogues.append(dialogue_data)
            if validate:
                print(f"  ✅ Validation passed: Answer {answer} matches GPT's calculation.")
            i += 1
        
        if attempts >= max_attempts:
            print(f"\n⚠️  WARNING: Reached maximum attempts ({max_attempts}) before generating {n} valid dialogues.")
        
        return dialogues
    
    def _validate_dialogue(self, dialogue: List[str]) -> int:
        """Validate a dialogue by asking GPT to calculate the answer."""
        question = dialogue[-1] if dialogue else ""
        facts = dialogue[:-1] if len(dialogue) > 1 else []
        
        prompt_parts = []
        if facts:
            prompt_parts.append("\n".join(facts))
        if question:
            prompt_parts.append(question)
        if INSTRUCTION:
            prompt_parts.append(INSTRUCTION)
        prompt = "\n".join(prompt_parts)
        
        try:
            llm = GetLlm(llm="gpt4", fallback=True)  # Always use GPT for validation
            content = llm.send(prompt)
            
            import re
            candidate = content.strip()
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
            if fence:
                candidate = fence.group(1).strip()
            
            try:
                result = json.loads(candidate)
                return int(result.get("answer", 0))
            except json.JSONDecodeError:
                m = re.search(r"-?\d+", candidate)
                return int(m.group(0)) if m else 0
        except Exception as e:
            print(f"  Error validating dialogue: {e}")
            return 0
    
    def _generate_with_gpt(self, problem_type: str) -> Tuple[List[str], int, str, str]:
        """Generate a dialogue using GPT in two steps."""
        core_facts, answer, justification = self._step1_generate_core(problem_type)
        dialogue = self._step2_apply_ofalma_factors(core_facts, problem_type)
        
        return dialogue, answer, justification, problem_type
    
    def _step1_generate_core(self, problem_type: str) -> Tuple[List[str], int, str]:
        """Step 1: Generate core facts for a problem with a known answer."""
        prompt = f"""Generate a natural, real-world {problem_type} problem with a known, calculable answer. Use concrete, tangible scenarios that feel authentic and conversational.

Requirements:
1. Create 5-7 core facts that are sufficient to calculate a single numeric answer
2. The answer must be an integer (e.g., days, hours, count, sum, etc.)
3. Facts should be clear and unambiguous, written in natural, fluent language
4. Use concrete scenarios (e.g., actual projects, real schedules, specific quantities, people's situations)
5. Avoid abstract labels like "Task X" or "Project Y" - use descriptive names or contexts
6. Make it feel like a real conversation or real-world scenario
7. Ensure the answer can be calculated from the facts
8. Provide a clear justification explaining why the answer is correct

Return a JSON object with:
- "facts": array of fact strings (5-7 facts) - written naturally, no markers like "CRITICAL" or "MUST"
- "question": string asking for the answer (format: "Given this information, what is X? Return only a single integer (the X). If you cannot determine it, return 0.")
- "answer": integer (the correct answer that can be calculated from the facts)
- "justification": string (explanation of why this answer is correct, showing the calculation/logic)

Example format:
{{
  "facts": ["The bakery needs 3 days to prepare all the ingredients for the wedding cake.", "Decorating the cake takes 2 days and can only start after ingredients are ready.", "The delivery setup takes 4 more days after decoration is complete."],
  "question": "Given this information, what is the minimum total number of days needed? Return only a single integer (the total days). If you cannot determine it, return 0.",
  "answer": 9,
  "justification": "Preparation takes 3 days. Decoration starts after preparation and takes 2 days, finishing on day 5. Delivery setup starts after decoration and takes 4 days, finishing on day 9. Therefore, the minimum total duration is 3 + 2 + 4 = 9 days."
}}"""
        
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        # Parse JSON from response
        import re
        candidate = content.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
        
        try:
            data = json.loads(candidate)
            facts = data.get("facts", [])
            question = data.get("question", "")
            answer = int(data.get("answer", 0))
            justification = data.get("justification", "No justification provided.")
            
            if not facts or answer == 0:
                raise ValueError("Invalid response")
            
            # Return core facts and question as a list, plus answer and justification
            core_dialogue = facts + [question]
            return core_dialogue, answer, justification
        except Exception as e:
            print(f"Error parsing Step 1 response: {e}")
            print(f"Content: {content[:200]}...")
            return ["Error generating core facts"], 0, "Error generating justification"
    
    def _step2_apply_ofalma_factors(self, core_dialogue: List[str], problem_type: str) -> List[str]:
        """Step 2: Apply OFALMA impact factors to create full dialogue."""
        core_facts = core_dialogue[:-1]  # All except question
        question = core_dialogue[-1]  # Last item is question
        
        prompt = f"""Given these core facts for a {problem_type} problem, expand the dialogue into a natural, fluent conversation that applies OFALMA impact factors organically.

Core facts (must be kept, placed early):
{chr(10).join(f"  - {fact}" for fact in core_facts)}

Create a natural dialogue that feels like a real conversation or documentation. The structure should naturally emphasize important information early and include less relevant details later, without explicit markers like "CRITICAL" or "MUST".

OFALMA Impact Factors (apply naturally, not explicitly):
- S (Surprisal): New/unexpected information
- R (Recency): Position in dialogue (later = higher)
- Q (Relevance/Clarity): Relevance to solving the problem
- E (Emphasis): Strength/assertiveness (achieved through natural language, not labels)

Create a full dialogue with:
1. Start with 2-4 important facts (high S, Q, E) - naturally emphasize the core facts through context, detail, or repetition. Write them in a way that feels important without labels.
2. Add 3-5 moderately relevant facts (mid Q, some E) - mention related processes, methods, or context that's somewhat relevant
3. Add 5-7 irrelevant facts (low Q, low S) at the end - include mundane, everyday details that are completely unrelated (coffee, parking, office amenities, weather, etc.)
4. DO NOT include the question in your response - it will be added separately
5. Write in natural, fluent language - no artificial markers or labels

Structure the dialogue so:
- Early facts (positions 1-4) should naturally convey high importance through language, detail, and context (Q=0.8-1.0, E=0.7-1.0)
- Middle facts (positions 5-8) should be moderately relevant (Q=0.4-0.7) - related processes or context
- Late facts (positions 9+) should be low relevance (Q=0.1-0.3) but recent - completely unrelated everyday details

Return a JSON object with:
- "dialogue": array of strings in order - written naturally, no explicit importance markers

Example (DO NOT include question - only facts, written naturally):
{{
  "dialogue": [
    "Sarah needs three full days to gather all the materials for the client presentation. This is the most time-consuming part of the whole project.",
    "Once the materials are ready, James will spend two days designing the slides. He can't start until Sarah finishes her part.",
    "The team uses a collaborative document system for tracking progress and sharing updates in real-time.",
    "Weekly review meetings are scheduled every Monday to discuss any blockers or adjustments needed.",
    "All design work goes through a peer review process before final approval.",
    "The office has a great coffee machine in the breakroom that everyone uses in the morning.",
    "Parking is always tight on Tuesdays when the farmers market is set up nearby.",
    "There's a nice view of the city park from the conference room windows.",
    "The building management recently updated the HVAC system for better air quality."
  ]
}}"""
        
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        # Parse JSON from response
        import re
        candidate = content.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
        
        try:
            data = json.loads(candidate)
            dialogue = data.get("dialogue", [])
            
            if not dialogue:
                raise ValueError("Empty dialogue")
            
            dialogue_cleaned = []
            question_found = False
            for item in dialogue:
                if not question_found and ("?" in item or "Return only" in item or "compute" in item.lower() or "calculate" in item.lower() or "what is" in item.lower() or "how many" in item.lower()):
                    continue
                dialogue_cleaned.append(item)
            dialogue_cleaned.append(question)
            
            return dialogue_cleaned
        except Exception as e:
            print(f"Error parsing Step 2 response: {e}")
            print(f"Content: {content[:200]}...")
            return core_dialogue + [
                "Additional context information.",
                "The team follows standard procedures.",
                "Office facilities are available.",
            ]


    def save_dataset(self, dialogues: List[Dict], filename: str = "dataset.json"):
        """Save generated dialogues to a JSON file."""
        import os
        if not os.path.isabs(filename) and not filename.startswith('data/'):
            filename = os.path.join('data', filename)
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'data', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(dialogues)} dialogues to {filename}")
    
    def load_dataset(self, filename: str = "dataset.json") -> List[Dict]:
        """Load dialogues from a JSON file."""
        import os
        if not os.path.isabs(filename) and not filename.startswith('data/'):
            filename = os.path.join('data', filename)
        with open(filename, 'r') as f:
            dialogues = json.load(f)
        print(f"Loaded {len(dialogues)} dialogues from {filename}")
        return dialogues


if __name__ == "__main__":
    import sys
    
    # Get n from command line or use default
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    generator = DataGenerator(seed=42)
    dialogues = generator.generate(n=n, validate=True)
    
    # Print first dialogue for review
    print("Generated Dialogue (First Example):")
    print("=" * 80)
    for dialogue_data in dialogues[:1]:
        print(f"\nID: {dialogue_data['id']}")
        print(f"Problem Type: {dialogue_data['problem_type']}")
        print(f"\nDialogue:")
        for i, utterance in enumerate(dialogue_data['dialogue'], 1):
            print(f"  {i:2d}. {utterance}")
        print(f"\nAnswer: {dialogue_data['answer']}")
        if 'justification' in dialogue_data:
            print(f"\nJustification:\n{dialogue_data['justification']}")
        print("=" * 80)
    
    # Save dataset
    filename = f"dataset_{n}.json"
    generator.save_dataset(dialogues, filename)
    
    print(f"\nGenerated {len(dialogues)} dialogues")
    print(f"Problem types: {[d['problem_type'] for d in dialogues]}")

