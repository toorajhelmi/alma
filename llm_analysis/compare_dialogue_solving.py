"""Compare how well Llama vs GPT solve dialogues from the dataset."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
from llm import get_provider, Gpt, Llama, TogetherAI, GetLlm
from langchain_openai import ChatOpenAI
from main import OPENAI_API_KEY, INSTRUCTION, parse_json_response

TOGETHER_API_KEY = "4164d6f1ac5f4fe46afde0cd1506316a50d700107897538642c83cd016b470a7"


def register_llms():
    """Register all available LLMs with the provider."""
    provider = get_provider()
    
    # Register GPT-4o
    gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    provider.register('gpt4o', Gpt(gpt_llm), default_remote=True)
    
    # Register GPT-4o-mini
    gpt_mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    provider.register('gpt4omini', Gpt(gpt_mini_llm), default_remote=False)
    print("Registered GPT-4o-mini")
    
    # Register Together.ai Llama 3.1 8B Instruct Turbo
    try:
        together_llm_8b = TogetherAI(api_key=TOGETHER_API_KEY, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        provider.register('llama31_8b', together_llm_8b, default_remote=False)
        print("Registered Together.ai Llama 3.1 8B Instruct Turbo")
    except Exception as e:
        print(f"Warning: Llama 3.1 8B not available: {e}")
    
    return True


def solve_dialogue_with_llm(dialogue: List[str], question: str, llm_name: str) -> Dict:
    """
    Solve a dialogue using a specific LLM.
    Uses the same approach as GPT in main.py: build prompt from facts + question + instruction.
    
    Args:
        dialogue: List of facts/utterances
        question: The question to answer
        llm_name: 'gpt4' or 'llama'
    
    Returns:
        Dictionary with answer, explanation, and correctness
    """
    # Build prompt exactly like main.py does
    final_prompt_parts = []
    if dialogue:
        final_prompt_parts.append("\n".join(dialogue))
    if question:
        final_prompt_parts.append(question)
    if INSTRUCTION:
        final_prompt_parts.append(INSTRUCTION)
    final_prompt = "\n".join(final_prompt_parts)
    
    # Get LLM and solve (same as main.py uses GetLlm)
    try:
        # Get LLM instance from provider
        llm = GetLlm(llm=llm_name, fallback=False)
        
        raw_content = llm.send(final_prompt)
        parsed = parse_json_response(raw_content)
        
        return {
            'llm': llm_name,
            'raw_response': raw_content,
            'parsed': parsed,
            'answer': parsed.get('answer') if isinstance(parsed, dict) else None,
            'explanation': parsed.get('explanation') if isinstance(parsed, dict) else None
        }
    except Exception as e:
        return {
            'llm': llm_name,
            'error': str(e),
            'answer': None
        }


def compare_on_dialogue(dialogue_data: Dict, expected_answer: int, model_names: List[str]) -> Dict:
    """
    Compare multiple models on solving a single dialogue.
    
    Args:
        dialogue_data: Dictionary with 'dialogue' array (last element is question)
        expected_answer: The correct answer
        model_names: List of model names to compare
    
    Returns:
        Comparison results
    """
    dialogue_list = dialogue_data.get('dialogue', [])
    
    # Last element is the question, rest are facts
    if len(dialogue_list) > 1:
        dialogue = dialogue_list[:-1]
        question = dialogue_list[-1]
    elif len(dialogue_list) == 1:
        dialogue = []
        question = dialogue_list[0]
    else:
        dialogue = []
        question = ''
    
    # Solve with each model
    model_results = {}
    for model_name in model_names:
        model_results[model_name] = solve_dialogue_with_llm(dialogue, question, model_name)
    
    # Check correctness for each model
    model_correct = {}
    for model_name in model_names:
        model_correct[model_name] = model_results[model_name].get('answer') == expected_answer
    
    # Build result dictionary
    result = {
        'dialogue_id': dialogue_data.get('id', 'unknown'),
        'expected_answer': expected_answer
    }
    
    # Add results for each model
    for model_name in model_names:
        result[model_name] = {
            'answer': model_results[model_name].get('answer'),
            'correct': model_correct[model_name],
            'explanation': model_results[model_name].get('explanation'),
            'error': model_results[model_name].get('error')
        }
    
    # Calculate agreement statistics
    all_correct = all(model_correct.values())
    all_wrong = not any(model_correct.values())
    
    result['all_correct'] = all_correct
    result['all_wrong'] = all_wrong
    
    # Find which models are correct/incorrect
    correct_models = [name for name, correct in model_correct.items() if correct]
    incorrect_models = [name for name, correct in model_correct.items() if not correct]
    
    result['correct_models'] = correct_models
    result['incorrect_models'] = incorrect_models
    
    return result


def analyze_dataset(num_samples: int = 5, model_names: List[str] = None):
    """
    Compare models on dialogues from the dataset.
    
    Args:
        num_samples: Number of dialogues to test
        model_names: List of model names to compare (e.g., ['gpt4o', 'gpt4omini', 'llama31_8b'])
    """
    if model_names is None:
        model_names = ['gpt4o', 'gpt4omini']
    
    dataset_file = "data/dataset_100.json"
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return
    
    # Load dataset
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample dialogues
    sample_size = min(num_samples, len(dataset))
    samples = dataset[:sample_size]
    
    model_names_str = " vs ".join(model_names)
    print(f"\n{'='*60}")
    print(f"Comparing {model_names_str} on {sample_size} dialogues from dataset")
    print(f"{'='*60}\n")
    
    results = []
    # Track correctness for each model
    model_correct_counts = {name: 0 for name in model_names}
    all_correct_count = 0
    all_wrong_count = 0
    
    for i, dialogue_data in enumerate(samples):
        dialogue_id = dialogue_data.get('id', i+1)
        print(f"\nProcessing {i+1}/{sample_size}: Dialogue ID {dialogue_id}")
        
        expected_answer = dialogue_data.get('answer')
        if expected_answer is None:
            print(f"  Skipping: No expected answer")
            continue
        
        if not dialogue_data.get('dialogue'):
            print(f"  Skipping: No dialogue data")
            continue
        
        result = compare_on_dialogue(dialogue_data, expected_answer, model_names)
        results.append(result)
        
        # Update statistics
        for model_name in model_names:
            if result[model_name]['correct']:
                model_correct_counts[model_name] += 1
        
        if result['all_correct']:
            all_correct_count += 1
        if result['all_wrong']:
            all_wrong_count += 1
        
        # Print result
        print(f"  Expected: {expected_answer}")
        for model_name in model_names:
            status = "✓" if result[model_name]['correct'] else "✗"
            answer = result[model_name]['answer']
            print(f"  {model_name:15} {status} Answer={answer}")
            if result[model_name].get('error'):
                print(f"    Error: {result[model_name]['error']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total dialogues: {sample_size}")
    print(f"\nAccuracy:")
    for model_name in model_names:
        correct = model_correct_counts[model_name]
        accuracy = correct / sample_size * 100
        print(f"  {model_name:15} {correct}/{sample_size} = {accuracy:.1f}%")
    
    print(f"\nAgreement:")
    print(f"  All correct:   {all_correct_count}/{sample_size} = {all_correct_count/sample_size*100:.1f}%")
    print(f"  All wrong:     {all_wrong_count}/{sample_size} = {all_wrong_count/sample_size*100:.1f}%")
    
    # Save results
    output_file = "llm_analysis/dialogue_solving_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        summary = {
            'total': sample_size,
            'models': model_names,
            'all_correct': all_correct_count,
            'all_wrong': all_wrong_count
        }
        for model_name in model_names:
            correct = model_correct_counts[model_name]
            summary[f'{model_name}_correct'] = correct
            summary[f'{model_name}_accuracy'] = correct / sample_size * 100
        
        json.dump({
            'summary': summary,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    print("Registering LLMs...")
    register_llms()
    
    # Specify models to compare by name
    models_to_compare = ['gpt4omini', 'llama31_8b']
    
    print(f"\nComparing {', '.join(models_to_compare)} on Dialogue Solving...")
    results = analyze_dataset(num_samples=10, model_names=models_to_compare)

