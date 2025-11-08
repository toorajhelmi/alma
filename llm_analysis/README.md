# LLM Analysis Scripts

This folder contains scripts to compare Llama vs GPT performance on OFALMA tasks.

## Scripts

### 1. `compare_impact_factors.py`

Compares how GPT and Llama compute impact factors (S, Q, E) for the same dialogues.

**Usage:**
```bash
python llm_analysis/compare_impact_factors.py
```

**Output:**
- Analyzes 5 sample dialogues by default
- Prints detailed comparison for each dialogue
- Shows mean differences for S, Q, E factors
- Saves results to `llm_analysis/impact_factors_comparison.json`

**Metrics:**
- Mean difference in Surprisal (S)
- Mean difference in Relevance/Clarity (Q)
- Mean difference in Emphasis (E)
- Overall average difference

### 2. `compare_dialogue_solving.py`

Compares how well GPT vs Llama solve dialogues from the dataset.

**Usage:**
```bash
python llm_analysis/compare_dialogue_solving.py
```

**Output:**
- Analyzes 5 dialogues from dataset by default
- Prints per-dialogue comparison
- Shows accuracy statistics for both LLMs
- Shows agreement/disagreement statistics
- Saves results to `llm_analysis/dialogue_solving_comparison.json`

**Metrics:**
- Accuracy: % of correct answers for GPT and Llama
- Agreement: How often both get the same result
- GPT-only correct: Cases where only GPT gets it right
- Llama-only correct: Cases where only Llama gets it right
- Both wrong: Cases where both fail

## Requirements

- Llama 3.1 must be installed via Ollama (`ollama pull llama3.1`)
- Dataset file at `data/dataset_100.json`
- All LLMs registered in main provider (done automatically by scripts)

## Output Files

- `impact_factors_comparison.json`: Detailed impact factor comparisons
- `dialogue_solving_comparison.json`: Dialogue solving accuracy results

