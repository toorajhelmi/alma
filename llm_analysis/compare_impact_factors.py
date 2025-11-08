"""Compare impact factors (S, Q, E) computed by Llama vs GPT."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
# Lazy imports to avoid langchain issues
# from core.ofalma import llm_impact_factors
from llm import get_provider, Gpt, Llama, TogetherAI
from langchain_openai import ChatOpenAI
from main import OPENAI_API_KEY

TOGETHER_API_KEY = "4164d6f1ac5f4fe46afde0cd1506316a50d700107897538642c83cd016b470a7"


def register_llms():
    """Register GPT and Together.ai Llama with the provider."""
    provider = get_provider()
    
    # Register GPT-4o
    gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    provider.register('gpt4', Gpt(gpt_llm), default_remote=True)
    
    # Register Together.ai Llama (much cheaper than GPT-4o)
    try:
        together_llm = TogetherAI(api_key=TOGETHER_API_KEY)
        provider.register('llama', together_llm, default_remote=False)
        print("Registered Together.ai Llama 3.1 8B Turbo")
        return True
    except Exception as e:
        print(f"Warning: Together.ai Llama not available: {e}")
        return False


def compare_impact_factors(dialogue: List[str]) -> Dict:
    """
    Compute impact factors for the same dialogue using both GPT and Llama.
    
    Returns:
        Dictionary with GPT and Llama impact factors, plus comparison metrics
    """
    # Import here to avoid circular imports
    from core.ofalma import llm_impact_factors as llm_impact_factors_fn
    import llm
    
    provider = get_provider()
    
    # Get GPT factors - directly use GPT LLM
    print("  Computing GPT impact factors...")
    import time
    start = time.time()
    gpt_factors = []
    try:
        gpt_llm = provider._llms['gpt4']
        
        # Patch GetLlm in both llm module and core.ofalma module
        import core.ofalma as ofalma_module
        original_get_llm_llm = llm.GetLlm
        original_get_llm_ofalma = ofalma_module.GetLlm
        
        def force_gpt(*args, **kwargs):
            return gpt_llm
        
        llm.GetLlm = force_gpt
        ofalma_module.GetLlm = force_gpt
        
        # GPT uses batch approach (per_turn=False) for stronger models
        gpt_factors = llm_impact_factors_fn(dialogue, verbose=False, per_turn=False)
        
        llm.GetLlm = original_get_llm_llm
        ofalma_module.GetLlm = original_get_llm_ofalma
        
        elapsed = time.time() - start
        print(f"  ✓ GPT factors computed: {len(gpt_factors)} factors in {elapsed:.1f}s")
        if gpt_factors and all(f.get('S', 0) == 0.5 and f.get('Q', 0) == 0.5 for f in gpt_factors):
            print("    ⚠ Warning: GPT returned default values - JSON parsing may have failed")
        
    except Exception as e:
        print(f"  ✗ Error computing GPT factors: {e}")
        import traceback
        traceback.print_exc()
    
    # Get Together.ai Llama factors
    print("  Computing Together.ai Llama impact factors...")
    start = time.time()
    llama_factors = []
    llama_available = False
    try:
        if 'llama' in provider._llms:
            llama_llm = provider._llms['llama']
            
            # Patch GetLlm in both llm module and core.ofalma module
            import core.ofalma as ofalma_module
            original_get_llm_llm = llm.GetLlm
            original_get_llm_ofalma = ofalma_module.GetLlm
            
            def force_llama(*args, **kwargs):
                return llama_llm
            
            llm.GetLlm = force_llama
            ofalma_module.GetLlm = force_llama
            
            # Llama uses per-turn approach (per_turn=True) for weaker models
            print("    → Using per-turn approach for Llama...")
            llama_factors = llm_impact_factors_fn(dialogue, verbose=False, per_turn=True)
            llama_available = True
            
            llm.GetLlm = original_get_llm_llm
            ofalma_module.GetLlm = original_get_llm_ofalma
            
            elapsed = time.time() - start
            print(f"  ✓ Together.ai Llama factors computed: {len(llama_factors)} factors in {elapsed:.1f}s")
            if llama_factors and all(f.get('S', 0) == 0.5 and f.get('Q', 0) == 0.5 for f in llama_factors):
                print("    ⚠ Warning: Llama returned default values - JSON parsing may have failed")
        else:
            print("  ✗ Llama not available in provider")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ Error computing Llama factors after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare factors
    comparison = {
        'dialogue': dialogue,
        'gpt_factors': gpt_factors,
        'llama_factors': llama_factors if llama_available else None,
        'llama_available': llama_available
    }
    
    if llama_available and llama_factors and len(gpt_factors) == len(llama_factors):
        # Calculate differences
        differences = []
        for i, (gpt_f, llama_f) in enumerate(zip(gpt_factors, llama_factors)):
            diff = {
                'index': i,
                'utterance': dialogue[i],
                'gpt': gpt_f,
                'llama': llama_f,
                'diff_S': abs(gpt_f.get('S', 0.5) - llama_f.get('S', 0.5)),
                'diff_Q': abs(gpt_f.get('Q', 0.5) - llama_f.get('Q', 0.5)),
                'diff_E': abs(gpt_f.get('E', 0.5) - llama_f.get('E', 0.5)),
                'avg_diff': (
                    abs(gpt_f.get('S', 0.5) - llama_f.get('S', 0.5)) +
                    abs(gpt_f.get('Q', 0.5) - llama_f.get('Q', 0.5)) +
                    abs(gpt_f.get('E', 0.5) - llama_f.get('E', 0.5))
                ) / 3.0
            }
            differences.append(diff)
        
        comparison['differences'] = differences
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        comparison['mean_diff_S'] = mean([d['diff_S'] for d in differences])
        comparison['mean_diff_Q'] = mean([d['diff_Q'] for d in differences])
        comparison['mean_diff_E'] = mean([d['diff_E'] for d in differences])
        comparison['mean_avg_diff'] = mean([d['avg_diff'] for d in differences])
    
    return comparison


def load_dialogues_from_dataset(num_samples: int = 20):
    """Load dialogues from dataset_100.json."""
    dataset_path = "data/dataset_100.json"
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dialogues = []
        for item in data[:num_samples]:
            # Extract facts (all dialogue items except the last one which is the question)
            dialogue = item['dialogue']
            # Remove the question (last item)
            facts = dialogue[:-1] if len(dialogue) > 1 else dialogue
            if facts:
                dialogues.append({
                    'id': item.get('id', len(dialogues) + 1),
                    'facts': facts,
                    'question': dialogue[-1] if len(dialogue) > 1 else None
                })
        return dialogues
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def analyze_sample_dialogues(num_samples: int = 20):
    """Compare impact factors on dialogues from dataset."""
    dialogues_data = load_dialogues_from_dataset(num_samples)
    
    if not dialogues_data:
        print("No dialogues loaded. Exiting.")
        return []
    
    num_available = len(dialogues_data)
    print(f"Loaded {num_available} dialogues from dataset")
    
    results = []
    import time
    total_start = time.time()
    
    for i, dialogue_data in enumerate(dialogues_data):
        dialogue = dialogue_data['facts']
        print(f"\n{'='*60}")
        print(f"Analyzing Dialogue {i+1}/{num_available} (ID: {dialogue_data['id']})")
        print(f"{'='*60}")
        print(f"Progress: {i}/{num_available} dialogues completed")
        dialogue_start = time.time()
        result = compare_impact_factors(dialogue)
        dialogue_time = time.time() - dialogue_start
        result['dialogue_id'] = dialogue_data['id']
        results.append(result)
        elapsed_total = time.time() - total_start
        avg_time = elapsed_total / (i + 1)
        remaining = avg_time * (num_available - i - 1)
        print(f"Dialogue {i+1} completed in {dialogue_time:.1f}s | Total: {elapsed_total:.1f}s | Est. remaining: {remaining:.1f}s")
    
    # Save results
    output_file = "llm_analysis/impact_factors_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    print("Registering LLMs...")
    llama_available = register_llms()
    
    if not llama_available:
        print("Cannot proceed: Llama not available")
        exit(1)
    
    print("\nComparing Impact Factors between GPT and Together.ai Llama...")
    results = analyze_sample_dialogues(num_samples=2)
    
    # Summary statistics
    if results and results[0].get('llama_available'):
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        def std(values):
            if not values:
                return 0.0
            m = mean(values)
            variance = sum((x - m) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        # Collect all GPT and Llama factors
        all_gpt_S = []
        all_gpt_Q = []
        all_gpt_E = []
        all_llama_S = []
        all_llama_Q = []
        all_llama_E = []
        all_diffs_S = []
        all_diffs_Q = []
        all_diffs_E = []
        
        for result in results:
            if result.get('llama_available') and 'gpt_factors' in result and 'llama_factors' in result:
                gpt_factors = result['gpt_factors']
                llama_factors = result['llama_factors']
                
                for gpt_f, llama_f in zip(gpt_factors, llama_factors):
                    gpt_s = gpt_f.get('S', 0.5)
                    gpt_q = gpt_f.get('Q', 0.5)
                    gpt_e = gpt_f.get('E', 0.5)
                    llama_s = llama_f.get('S', 0.5)
                    llama_q = llama_f.get('Q', 0.5)
                    llama_e = llama_f.get('E', 0.5)
                    
                    all_gpt_S.append(gpt_s)
                    all_gpt_Q.append(gpt_q)
                    all_gpt_E.append(gpt_e)
                    all_llama_S.append(llama_s)
                    all_llama_Q.append(llama_q)
                    all_llama_E.append(llama_e)
                    all_diffs_S.append(abs(gpt_s - llama_s))
                    all_diffs_Q.append(abs(gpt_q - llama_q))
                    all_diffs_E.append(abs(gpt_e - llama_e))
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total utterances analyzed: {len(all_gpt_S)}")
        print(f"\nGPT-4o Impact Factors:")
        print(f"  S (Surprisal):     {mean(all_gpt_S):.3f} ± {std(all_gpt_S):.3f}")
        print(f"  Q (Relevance):     {mean(all_gpt_Q):.3f} ± {std(all_gpt_Q):.3f}")
        print(f"  E (Emphasis):      {mean(all_gpt_E):.3f} ± {std(all_gpt_E):.3f}")
        print(f"\nTogether.ai Llama 3.1 8B Impact Factors:")
        print(f"  S (Surprisal):     {mean(all_llama_S):.3f} ± {std(all_llama_S):.3f}")
        print(f"  Q (Relevance):     {mean(all_llama_Q):.3f} ± {std(all_llama_Q):.3f}")
        print(f"  E (Emphasis):      {mean(all_llama_E):.3f} ± {std(all_llama_E):.3f}")
        print(f"\nDifferences (GPT vs Llama):")
        print(f"  S (Surprisal):     {mean(all_diffs_S):.3f} ± {std(all_diffs_S):.3f}")
        print(f"  Q (Relevance):     {mean(all_diffs_Q):.3f} ± {std(all_diffs_Q):.3f}")
        print(f"  E (Emphasis):      {mean(all_diffs_E):.3f} ± {std(all_diffs_E):.3f}")
        print(f"  Overall Average:   {(mean(all_diffs_S) + mean(all_diffs_Q) + mean(all_diffs_E))/3:.3f}")
    else:
        print("\nCannot compute statistics: Llama not available or no results")

