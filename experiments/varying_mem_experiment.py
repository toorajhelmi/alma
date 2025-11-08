"""Experiment for comparing approaches under varying memory limits."""

from __future__ import annotations

import json
import os
import sys
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
    run_ofalma,
    run_ofalma_rate_distortion,
    run_oneshot_from_memory,
)


@dataclass
class DialogueExample:
    """Structured dialogue sample with expected answer."""

    id: Optional[int]
    facts: List[str]
    question: str
    answer: Optional[int]


def load_dialogues_from_dataset(path: str, limit: int) -> List[DialogueExample]:
    """Load dialogue examples from dataset JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dialogues: List[DialogueExample] = []
    for item in data[:limit]:
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
    num_dialogues: int = 3,
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

    if proportions is None:
        proportions = (0.25, 0.5, 0.75, 1.0)

    dialogues = load_dialogues_from_dataset(dataset_path, num_dialogues)
    if not dialogues:
        print("No dialogues loaded for experiment.")
        return

    ensure_evaluation_llm_registered(evaluation_model)

    eval_llm = GetLlm(llm=evaluation_model, fallback=True) if evaluation_model else GetLlm(fallback=True)
    token_fn = eval_llm.get_token_count_fn()

    fact_token_counts = [sum(token_fn(fact) for fact in dialogue.facts) for dialogue in dialogues]
    avg_fact_tokens = sum(fact_token_counts) / len(fact_token_counts)

    print("\n=== Memory Limit Experiment ===")
    print(f"Dialogues: {len(dialogues)}")
    print(f"Average fact tokens: {avg_fact_tokens:.1f}")

    approaches = [
        "token_buffer",
        "summary",
        "custom_summary",
        "ofalma",
        "ofalma_rate_distortion",
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

    for prop in proportions:
        memory_limit = max(1, int(round(avg_fact_tokens * prop)))
        print(f"\n--- Memory ratio {prop:.2f} (limit = {memory_limit} tokens) ---")

        for dialogue in dialogues:
            facts = dialogue.facts
            question = dialogue.question
            expected_answer = dialogue.answer

            for approach in approaches:
                outcome = None

                try:
                    if approach in {"token_buffer", "summary", "custom_summary"}:
                        chain_result = create_conversational_chain(
                            memory_type=approach if approach != "token_buffer" else "token_buffer",
                            buffer_size=memory_limit,
                            model_name=evaluation_model or "gpt-4o",
                            verbose=False,
                            memory_verbose=False,
                        )

                        if approach == "custom_summary":
                            chain, memory = chain_result
                            outcome = run_oneshot_from_memory(
                                None,
                                facts,
                                question,
                                evaluation_model=evaluation_model,
                                memory=memory,
                            )
                        else:
                            chain = chain_result
                            outcome = run_oneshot_from_memory(
                                chain,
                                facts,
                                question,
                                evaluation_model=evaluation_model,
                            )

                    elif approach == "ofalma":
                        outcome = run_ofalma(
                            facts,
                            question,
                            buffer_size=memory_limit,
                            verbose=False,
                            impact_model=impact_model,
                            token_model=token_model,
                            evaluation_model=evaluation_model,
                            display=False,
                        )

                    elif approach == "ofalma_rate_distortion":
                        outcome = run_ofalma_rate_distortion(
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
                except Exception as exc:  # pylint: disable=broad-except
                    warning_msg = (
                        f"    [WARN] {approach} failed for dialogue {dialogue.id} "
                        f"at ratio {prop:.2f}: {exc}"
                    )
                    print(warning_msg)
                    failures.append(
                        {
                            "dialogue_id": dialogue.id,
                            "approach": approach,
                            "memory_ratio": prop,
                            "memory_limit": memory_limit,
                            "error": str(exc),
                        }
                    )
                    continue

                if not outcome:
                    continue

                # Consistent console output for every approach.
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

                if (
                    answer_value is not None
                    and expected_answer is not None
                    and answer_value == expected_answer
                ):
                    approach_stats["correct"] += 1

                run_records.append(
                    {
                        "dialogue_id": dialogue.id,
                        "approach": approach,
                        "memory_ratio": prop,
                        "memory_limit": memory_limit,
                        "prompt_tokens": prompt_tokens,
                        "expected_answer": expected_answer,
                        "predicted_answer": answer_value,
                        "correct": bool(
                            answer_value is not None
                            and expected_answer is not None
                            and answer_value == expected_answer
                        ),
                    }
                )

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

    # Persist experiment results for offline analysis / plotting.
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        output_file = Path(output_path)
        if output_file.suffix.lower() != ".json":
            output_file = output_file.with_suffix(".json")
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = log_dir_path / f"varying_mem_experiment_{timestamp}.json"

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

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)

    print(f"\nExperiment log saved to {output_file}")
    return output_file


if __name__ == "__main__":
    run_varying_memory_experiment()

