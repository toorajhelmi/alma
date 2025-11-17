"""Experiment for comparing approaches under varying memory limits."""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from langchain_openai import ChatOpenAI  # pylint: disable=wrong-import-position

from llm import GetLlm, Gpt, get_provider  # pylint: disable=wrong-import-position

# Reuse existing helpers from main module.
from main import (  # type: ignore  # pylint: disable=wrong-import-position
    OPENAI_API_KEY,
    create_conversational_chain,
    print_results,
    run_falma,
    run_falma_rate_distortion,
    run_oneshot_from_memory,
)
from core.falma import theta as FALMA_THETA  # type: ignore  # pylint: disable=wrong-import-position


@dataclass
class DialogueExample:
    """Structured dialogue sample with expected answer."""

    id: Optional[int]
    facts: List[str]
    question: str
    answer: Optional[int]


def load_dialogues_from_dataset(path: str, limit: int, start_index: int = 0) -> List[DialogueExample]:
    """Load dialogue examples from dataset JSON file.
    
    Args:
        path: Path to dataset JSON file
        limit: Maximum number of dialogues to load
        start_index: Starting index in the dataset (default: 0)
    """

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dialogues: List[DialogueExample] = []
    end_index = min(start_index + limit, len(data))
    for item in data[start_index:end_index]:
        dialogue_lines: Sequence[str] = item.get("dialogue", [])
        if len(dialogue_lines) < 2:
            continue

        facts = list(dialogue_lines[:-1])
        question = dialogue_lines[-1]
        dialogues.append(
            DialogueExample(
                id=item.get("id"),
                facts=facts,
                question=question,
                answer=item.get("answer"),
            )
        )

    return dialogues


def ensure_evaluation_llm_registered(model_name: str) -> None:
    """Ensure the requested evaluation model is registered with the provider."""

    provider = get_provider()

    target = model_name or "gpt-4o"
    if provider.is_available(target):
        return

    chat = ChatOpenAI(model=target, temperature=0, openai_api_key=OPENAI_API_KEY)
    provider.register(target, Gpt(chat), default_remote=True)

    # Aliases for compatibility (some callers expect "gpt4")
    if target != "gpt4" and not provider.is_available("gpt4"):
        provider.register("gpt4", Gpt(chat), default_remote=False)


def run_varying_memory_experiment(
    num_dialogues: int = 20,
    proportions: Optional[Sequence[float]] = None,
    dataset_path: str = "data/dataset_100.json",
    evaluation_model: str = "gpt-4o",
    impact_model: Optional[str] = None,
    token_model: Optional[str] = None,
    condensation_model: Optional[str] = None,
    output_path: Optional[str] = None,
    log_dir: str = "experiments_logs",
) -> Optional[Path]:
    """Execute memory limit experiment across approaches."""

    # We focus on condensed cases only: 25%, 50%, 75% of the average fact tokens.
    # There is no benefit to running a 100% ratio here (no condensation needed).
    if proportions is None:
        proportions = (0.25, 0.5, 0.75)

    # Load from validation set (indices 80-99) to avoid bias
    # Training used indices 0-79, validation used 80-99 (80/20 split)
    validation_start_index = 80
    dialogues = load_dialogues_from_dataset(dataset_path, num_dialogues, start_index=validation_start_index)
    if not dialogues:
        print("No dialogues loaded for experiment.")
        return
    
    print(f"Loading {len(dialogues)} dialogues from validation set (indices {validation_start_index}-{validation_start_index + len(dialogues) - 1})")

    ensure_evaluation_llm_registered(evaluation_model)

    eval_llm = GetLlm(llm=evaluation_model, fallback=True) if evaluation_model else GetLlm(fallback=True)
    token_fn = eval_llm.get_token_count_fn()

    fact_token_counts = [sum(token_fn(fact) for fact in dialogue.facts) for dialogue in dialogues]
    avg_fact_tokens = sum(fact_token_counts) / len(fact_token_counts)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if output_path:
        output_file = Path(output_path)
        if output_file.suffix.lower() != ".json":
            output_file = output_file.with_suffix(".json")
        base_name = output_file.stem
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"varying_mem_experiment_{timestamp}"
        output_file = log_dir_path / f"{base_name}.json"

    incremental_log_path = log_dir_path / f"{base_name}.jsonl"
    if incremental_log_path.exists():
        incremental_log_path.unlink()

    log_lock = threading.Lock()

    def make_json_safe(value):
        if isinstance(value, dict):
            return {k: make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(v) for v in value]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)) or value is None:
            return value
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return item_method()
            except Exception:
                pass
        return str(value)

    def append_log(entry: Dict[str, object]) -> None:
        payload = make_json_safe(dict(entry))
        payload.setdefault("timestamp_utc", datetime.utcnow().isoformat())
        with log_lock:
            with incremental_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(payload))
                log_file.write("\n")

    parameters_payload = {
        "num_dialogues": num_dialogues,
        "proportions": list(proportions),
        "dataset_path": dataset_path,
        "evaluation_model": evaluation_model,
        "impact_model": impact_model,
        "token_model": token_model,
        "condensation_model": condensation_model,
    }

    print("\n=== Memory Limit Experiment ===")
    print(f"Dialogues: {len(dialogues)}")
    print(f"Average fact tokens: {avg_fact_tokens:.1f}")
    print(f"Streaming incremental logs to {incremental_log_path}")

    append_log(
        {
            "type": "start",
            "parameters": parameters_payload,
            "average_fact_tokens": avg_fact_tokens,
        }
    )

    approaches = [
        "token_buffer",
        "summary",
        "custom_summary",
        "falma",
        "falma_rate_distortion",
    ]

    results: Dict[str, Dict[float, Dict[str, float]]] = {
        approach: {
            prop: {
                "correct": 0,
                "runs": 0,
                "prompt_tokens": 0,
            }
            for prop in proportions
        }
        for approach in approaches
    }

    run_records: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    # Pre-computed, ratio-specific FALMA weights learned via PPO RL (pruning method)
    # 75% memory (buffer ≈ 260 tokens): best weights at step 10k (100% val, 87% train)
    # 50% memory (buffer ≈ 180 tokens): best weights at step 10k (100% val, 93% train)
    # 25% memory (buffer ≈ 90 tokens): best weights at step 10k (70% val, 54% train)
    ratio_specific_theta = {
        0.75: {"S": 0.1735, "R": 0.0044, "Q": 0.7763, "E": 0.5565},
        0.50: {"S": 0.0167, "R": 0.0000, "Q": 0.7390, "E": 0.4602},
        0.25: {"S": 0.0099, "R": 0.0096, "Q": 0.2555, "E": 0.3080},
    }

    for prop in proportions:
        memory_limit = max(1, int(round(avg_fact_tokens * prop)))
        print(f"\n--- Memory ratio {prop:.2f} (limit = {memory_limit} tokens) ---")

        # Update FALMA global theta to the ratio-specific trained weights, if available.
        # This affects both pruning (`falma`) and rate-distortion (`falma_rate_distortion`)
        # since they both compute importance using the global `theta` in `core.falma`.
        # We round the proportion to two decimals to match the keys above.
        rounded_prop = round(float(prop), 2)
        if rounded_prop in ratio_specific_theta:
            weights = ratio_specific_theta[rounded_prop]
            FALMA_THETA["S"] = weights["S"]
            FALMA_THETA["R"] = weights["R"]
            FALMA_THETA["Q"] = weights["Q"]
            FALMA_THETA["E"] = weights["E"]
            print(
                f"  Using FALMA weights for ratio {rounded_prop:.2f}: "
                f"S={weights['S']:.3f}, R={weights['R']:.3f}, "
                f"Q={weights['Q']:.3f}, E={weights['E']:.3f}"
            )

        for dialogue in dialogues:
            facts = dialogue.facts
            question = dialogue.question
            expected_answer = dialogue.answer

            def run_single_approach(approach_name: str):
                try:
                    if approach_name in {"token_buffer", "summary", "custom_summary"}:
                        chain_result = create_conversational_chain(
                            memory_type=approach_name,
                            buffer_size=memory_limit,
                            model_name=evaluation_model or "gpt-4o",
                            verbose=False,
                            memory_verbose=False,
                        )

                        if approach_name == "custom_summary":
                            chain, memory = chain_result
                            outcome_local = run_oneshot_from_memory(
                                None,
                                facts,
                                question,
                                evaluation_model=evaluation_model,
                                memory=memory,
                            )
                        else:
                            chain = chain_result
                            outcome_local = run_oneshot_from_memory(
                                chain,
                                facts,
                                question,
                                evaluation_model=evaluation_model,
                            )

                    elif approach_name == "falma":
                        outcome_local = run_falma(
                            facts,
                            question,
                            buffer_size=memory_limit,
                            verbose=False,
                            impact_model=impact_model,
                            token_model=token_model,
                            evaluation_model=evaluation_model,
                            display=False,
                        )

                    elif approach_name == "falma_rate_distortion":
                        outcome_local = run_falma_rate_distortion(
                            facts,
                            question,
                            buffer_size=memory_limit,
                            verbose=False,
                            k=3.0,
                            condensation_model=condensation_model,
                            impact_model=impact_model,
                            token_model=token_model,
                            evaluation_model=evaluation_model,
                            display=False,
                        )
                    else:
                        raise ValueError(f"Unknown approach '{approach_name}'")

                    return {"approach": approach_name, "outcome": outcome_local, "error": None}

                except Exception as exc:  # pylint: disable=broad-except
                    return {"approach": approach_name, "outcome": None, "error": exc}

            with ThreadPoolExecutor(max_workers=len(approaches)) as executor:
                future_to_approach = {
                    executor.submit(run_single_approach, approach): approach for approach in approaches
                }
                results_batch = []
                for future in as_completed(future_to_approach):
                    results_batch.append(future.result())

            ordered_results = {item["approach"]: item for item in results_batch}

            for approach in approaches:
                result_info = ordered_results.get(approach)
                if not result_info:
                    continue

                error = result_info["error"]
                outcome = result_info["outcome"]

                if error:
                    warning_msg = (
                        f"    [WARN] {approach} failed for dialogue {dialogue.id} "
                        f"at ratio {prop:.2f}: {error}"
                    )
                    print(warning_msg)
                    append_log(
                        {
                            "type": "failure",
                            "dialogue_id": dialogue.id,
                            "approach": approach,
                            "memory_ratio": prop,
                            "memory_limit": memory_limit,
                            "error": str(error),
                        }
                    )
                    failures.append(
                        {
                            "dialogue_id": dialogue.id,
                            "approach": approach,
                            "memory_ratio": prop,
                            "memory_limit": memory_limit,
                            "error": str(error),
                        }
                    )
                    continue

                if not outcome:
                    continue

                print(
                    f"\n=== Approach: {approach} | Dialogue: {dialogue.id} | "
                    f"Memory ratio: {prop:.2f} (limit={memory_limit}) ==="
                )
                print_results(outcome)

                stats = outcome.get("stats", {})
                prompt_tokens = stats.get("final_prompt_tokens", 0) or 0

                answer_value = None
                result_dict = outcome.get("result")
                if isinstance(result_dict, dict):
                    answer_value = result_dict.get("answer")

                approach_stats = results[approach][prop]
                approach_stats["runs"] += 1
                approach_stats["prompt_tokens"] += prompt_tokens

                is_correct = (
                    answer_value is not None
                    and expected_answer is not None
                    and answer_value == expected_answer
                )

                if is_correct:
                    approach_stats["correct"] += 1

                run_record = {
                    "dialogue_id": dialogue.id,
                    "approach": approach,
                    "memory_ratio": prop,
                    "memory_limit": memory_limit,
                    "prompt_tokens": prompt_tokens,
                    "expected_answer": expected_answer,
                    "predicted_answer": answer_value,
                    "correct": bool(is_correct),
                }

                run_records.append(run_record)
                append_log({"type": "run", **run_record})

    print("\n=== Experiment Summary ===")
    for prop in proportions:
        memory_limit = max(1, int(round(avg_fact_tokens * prop)))
        print(f"\nMemory ratio {prop:.2f} (limit = {memory_limit})")
        for approach in approaches:
            data = results[approach][prop]
            runs = data["runs"] or 1
            avg_tokens = data["prompt_tokens"] / runs
            print(
                f"  {approach}: correct {int(data['correct'])}/{int(data['runs'])}, "
                f"avg prompt tokens {avg_tokens:.1f}"
            )

    summary_payload = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "parameters": {
            "num_dialogues": num_dialogues,
            "proportions": list(proportions),
            "dataset_path": dataset_path,
            "evaluation_model": evaluation_model,
            "impact_model": impact_model,
            "token_model": token_model,
            "condensation_model": condensation_model,
        },
        "average_fact_tokens": avg_fact_tokens,
        "summary": {
            approach: {
                str(prop): {
                    "correct": int(data["correct"]),
                    "runs": int(data["runs"]),
                    "avg_prompt_tokens": (
                        (data["prompt_tokens"] / data["runs"])
                        if data["runs"]
                        else 0.0
                    ),
                }
                for prop, data in prop_map.items()
            }
            for approach, prop_map in results.items()
        },
        "runs": run_records,
        "failures": failures,
    }

    safe_summary_payload = make_json_safe(summary_payload)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(safe_summary_payload, file, indent=2)

    append_log({"type": "summary", "summary": safe_summary_payload["summary"]})

    print(f"\nExperiment log saved to {output_file}")
    return output_file


if __name__ == "__main__":
    run_varying_memory_experiment()

