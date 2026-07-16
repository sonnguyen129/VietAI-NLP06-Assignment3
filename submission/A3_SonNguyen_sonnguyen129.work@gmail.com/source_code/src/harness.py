"""
harness.py — Full EvoAgent training loop.

Orchestrates T iterations of:
  1. Propose a new strategy (or use the seed for iteration 0).
  2. Evaluate on the train subset.
  3. Evaluate on the dev split.
  4. Reflect on the results.
  5. Save state to disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from tqdm import tqdm

from src.executor import EvalResult, evaluate, classify_question_type, TokenBudget
from src.model import QwenInference
from src.self_proposer import propose_self
from src.self_reflector import reflect_self
from src.strategy import CoTFormat, Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy

logger = logging.getLogger(__name__)


def select_parent_strategy(
    history: StrategyHistory,
    afo_mode: str,
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
) -> Optional[Strategy]:
    """
    Select the parent strategy to mutate based on the Always-From-Original (AFO) mode.

    TODO: Implement the Always-From-Original (AFO) parent selection policy.
    Modes:
    - 'none': Always mutate from the latest strategy in history.
    - 'best': Always mutate from the best strategy (highest dev accuracy) found so far.
    - 'original': Always mutate from the original seed strategy (iteration 0).
    - 'probabilistic': Select from Best, Original, and Latest based on the provided probabilities.
    """
    if not history.strategies:
        return None

    latest = history.latest_strategy()
    original = history.strategies[0]
    best = history.best_strategy() or latest

    if afo_mode == "none":
        return latest
    if afo_mode == "best":
        return best
    if afo_mode == "original":
        return original
    if afo_mode == "probabilistic":
        import random

        return random.choices(
            [best, original, latest],
            weights=[afo_prob_best, afo_prob_original, afo_prob_latest],
            k=1,
        )[0]

    logger.warning("Unknown afo_mode %r — falling back to latest strategy.", afo_mode)
    return latest


def select_curriculum_dataset(train_dataset: Dataset, iteration: int, train_size: int) -> Dataset:
    """
    Selects the train_subset dynamically based on curriculum learning.
    - Iteration 1: Easiest, shortest reading passages.
    - Iteration 2: Harder "table_op" or multi-step questions.
    - Other iterations (like 0, 3+): Easiest or standard subset.
    """
    rows = list(train_dataset)
    row_details = []
    for r in rows:
        passage = r.get("context") or r.get("article") or r.get("passage") or ""
        gold_program = r.get("answer") or ""
        
        q_type = classify_question_type(gold_program)
        is_hard = q_type in ["table_op", "division"] or ("," in gold_program)
        
        row_details.append({
            "row": r,
            "passage_len": len(passage),
            "is_hard": is_hard
        })

    if iteration == 2:
        # Prioritize rows with harder questions, then sort by passage length descending
        row_details.sort(key=lambda x: (0 if x["is_hard"] else 1, -x["passage_len"]))
    else:
        # Prioritize rows with easier questions, then sort by passage length ascending
        row_details.sort(key=lambda x: (1 if x["is_hard"] else 0, x["passage_len"]))

    selected_rows = [x["row"] for x in row_details[:min(train_size, len(row_details))]]
    return Dataset.from_list(selected_rows)


def run_smoke_test(
    strategy: Strategy,
    train_dataset: Dataset,
    model: QwenInference,
) -> bool:
    """
    Run the strategy on up to 5 examples to verify structural validity.
    Returns True if it passes, False if it fails.

    TODO: Implement the pre-flight smoke test.
    Steps:
      1. Select up to 5 examples from train_dataset.
      2. Temporarily set model.max_new_tokens based on strategy.cot_format (4096 if CoT, 256 if direct).
      3. Evaluate the strategy on this subset using evaluate().
      4. Check that at least one predicted answer is not None (i.e. program extraction succeeded).
      5. Check that average output tokens generated per question does not exceed 90% of the token limit 
         (to prevent infinite looping/truncation).
      6. Return True if valid, False otherwise. Make sure to restore model.max_new_tokens at the end.
    """
    subset = train_dataset.select(range(min(5, len(train_dataset))))
    limit = 4096 if strategy.cot_format != CoTFormat.NONE else 256

    original_max_new_tokens = model.max_new_tokens
    model.max_new_tokens = limit
    try:
        result = evaluate(strategy, "smoke_test", subset, model)
    except Exception as exc:
        logger.warning("Smoke test crashed: %s", exc)
        return False
    finally:
        model.max_new_tokens = original_max_new_tokens

    if not result.per_question:
        logger.warning("Smoke test failed: no predictions returned.")
        return False

    num_extracted = sum(1 for r in result.per_question if r.predicted_answer is not None)
    if num_extracted == 0:
        logger.warning("Smoke test failed: no valid program extracted from any output.")
        return False

    avg_output_tokens = sum(r.output_tokens for r in result.per_question) / len(result.per_question)
    if avg_output_tokens >= 0.9 * limit:
        logger.warning(
            "Smoke test failed: avg output tokens %.1f >= 90%% of limit %d (truncation risk).",
            avg_output_tokens, limit,
        )
        return False

    logger.info(
        "Smoke test passed: %d/%d programs extracted, avg output tokens %.1f (limit %d).",
        num_extracted, len(result.per_question), avg_output_tokens, limit,
    )
    return True


def run_evoagent(
    T: int,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: QwenInference,
    output_dir: Path,
    train_size: int = 100,
    resume_from: Optional[Path] = None,
    early_stop_accuracy: float = 1.0,
    afo_mode: str = "probabilistic",
    afo_prob_best: float = 0.4,
    afo_prob_original: float = 0.3,
    afo_prob_latest: float = 0.3,
    progressive_reflections: bool = True,
    use_curriculum: bool = False,
) -> StrategyHistory:
    """
    Run the EvoAgent loop for up to T iterations.

    TODO: Implement the EvoAgent optimization loop.
    For each iteration (from start_iteration up to T-1):
      1. Propose strategy:
         - Iteration 0: Use make_seed_strategy()
         - Iterations > 0: Mutate from a parent selected via select_parent_strategy().
           Try proposing up to 3 times, validating each using run_smoke_test().
      2. Set model.max_new_tokens dynamically (4096 if CoT, 256 if direct).
      3. Evaluate on train subset (curriculum or slice) and dev split.
      4. Accumulate token usage in TokenBudget.
      5. Reflect on errors (except on the last iteration T-1).
      6. Append/save strategies, evaluations, and reflections to the history file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.jsonl"
    history = StrategyHistory(history_path)

    if resume_from is not None:
        history.path = Path(resume_from)
        history.load()
        logger.info("Resumed from %s — %d strategies in history.", resume_from, len(history))
    elif history_path.exists():
        history.load()
        logger.info("Auto-resumed from %s — %d strategies in history.", history_path, len(history))

    budget = TokenBudget()
    start_iteration = len(history.strategies)

    if start_iteration >= T:
        logger.info("History already has %d strategies (T=%d). Nothing to do.", start_iteration, T)
        return history

    logger.info("Starting EvoAgent loop. Iterations %d–%d (T=%d).", start_iteration, T - 1, T)

    for iteration in range(start_iteration, T):
        iter_start = time.time()
        is_last_iteration = iteration == T - 1
        logger.info("=== Iteration %d/%d ===", iteration, T - 1)

        # ------------------------------------------------------------------
        # 1. Obtain the strategy for this iteration.
        # ------------------------------------------------------------------
        propose_tokens = 0
        if iteration == 0:
            strategy = make_seed_strategy()
        else:
            parent = select_parent_strategy(
                history,
                afo_mode=afo_mode,
                afo_prob_best=afo_prob_best,
                afo_prob_original=afo_prob_original,
                afo_prob_latest=afo_prob_latest,
            )
            logger.info(
                "AFO parent (%s): %s", afo_mode, parent.id[:8] if parent else "None"
            )

            strategy = None
            last_candidate = None
            for attempt in range(1, 4):
                candidate, tokens = propose_self(
                    history,
                    model,
                    max_retries=3,
                    parent_strategy_id=parent.id if parent else None,
                    train_dataset=train_dataset,
                )
                propose_tokens += tokens
                last_candidate = candidate
                if run_smoke_test(candidate, train_dataset, model):
                    strategy = candidate
                    break
                logger.warning(
                    "Proposal attempt %d/3 failed the smoke test — re-proposing.", attempt
                )
            if strategy is None:
                # All proposals failed the smoke test: accept the last candidate
                # anyway so the loop keeps evolving instead of stalling.
                logger.warning(
                    "All 3 proposals failed the smoke test — accepting the last one."
                )
                strategy = last_candidate

        strategy.metadata.iteration = iteration

        # ------------------------------------------------------------------
        # 2. Decode cap per strategy: CoT needs room, direct answers do not.
        # ------------------------------------------------------------------
        model.max_new_tokens = 4096 if strategy.cot_format != CoTFormat.NONE else 256

        # ------------------------------------------------------------------
        # 3. Evaluate on train subset (curriculum-aware) and dev split.
        # ------------------------------------------------------------------
        if use_curriculum:
            train_subset = select_curriculum_dataset(train_dataset, iteration, train_size)
        else:
            train_subset = train_dataset.select(
                range(min(train_size, len(train_dataset)))
            )

        history.append_strategy(strategy)
        _save_strategy_json(strategy, output_dir, iteration)

        train_result = evaluate(strategy, "train", train_subset, model)
        _save_eval_result(train_result, output_dir, iteration, "train")
        logger.info("Train accuracy: %.3f", train_result.accuracy)

        dev_result = evaluate(strategy, "dev", dev_dataset, model)
        _save_eval_result(dev_result, output_dir, iteration, "dev")
        logger.info("Dev accuracy: %.3f", dev_result.accuracy)

        # ------------------------------------------------------------------
        # 4. Token accounting.
        # ------------------------------------------------------------------
        budget.add_eval(train_result)
        budget.add_eval(dev_result)

        # ------------------------------------------------------------------
        # 5. Reflect (skipped on the final iteration — nothing left to propose).
        # ------------------------------------------------------------------
        reflect_tokens = 0
        if not is_last_iteration:
            reflection, reflect_tokens = reflect_self(
                strategy,
                dev_result,
                model,
                progressive=progressive_reflections,
            )
            history.append_reflection(reflection)
            _save_reflection_json(reflection, output_dir, iteration)

        budget.add_meta(propose_tokens + reflect_tokens)

        # ------------------------------------------------------------------
        # 6. Persist metadata and report progress.
        # ------------------------------------------------------------------
        metadata = strategy.metadata
        metadata.train_accuracy = train_result.accuracy
        metadata.dev_accuracy = dev_result.accuracy
        metadata.token_cost_claude = propose_tokens + reflect_tokens
        metadata.token_cost_qwen = (
            train_result.total_input_tokens
            + train_result.total_output_tokens
            + dev_result.total_input_tokens
            + dev_result.total_output_tokens
        )
        history.update_strategy_metadata(strategy.id, metadata)

        logger.info(
            "Iteration %d done in %.1fs. %s",
            iteration,
            time.time() - iter_start,
            budget.summary(),
        )
        _print_leaderboard(history)

        if dev_result.accuracy >= early_stop_accuracy:
            logger.info(
                "Early stop: dev accuracy %.3f >= %.3f.",
                dev_result.accuracy,
                early_stop_accuracy,
            )
            break

    return history


# ------------------------------------------------------------------
# Internal Helpers for saving results
# ------------------------------------------------------------------

def _save_strategy_json(strategy: Strategy, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"iter_{iteration:03d}_strategy.json"
    path.write_text(strategy.to_json(), encoding="utf-8")


def _save_eval_result(
    result: EvalResult,
    output_dir: Path,
    iteration: int,
    tag: str,
) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_eval_{tag}.json"
    data = result.to_dict()
    data["per_question"] = [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "gold_val": r.gold_val,
            "predicted_answer": r.predicted_answer,
            "predicted_val": r.predicted_val,
            "is_correct": r.is_correct,
            "question_type": r.question_type,
            "raw_output": r.raw_output,
        }
        for r in result.per_question
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_reflection_json(reflection, output_dir: Path, iteration: int) -> None:
    import json
    path = output_dir / f"iter_{iteration:03d}_reflection.json"
    path.write_text(
        json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_leaderboard(history: StrategyHistory) -> None:
    rows = history.summary_table()
    if not rows:
        return
    header = f"{'Iter':>4}  {'ID':>8}  {'CoT':>10}  {'Dev Acc':>8}  {'Train Acc':>9}  {'Meta tok':>10}  {'Qwen tok':>8}"
    logger.info("Leaderboard:\n%s", header)
    for r in rows:
        dev = f"{r['dev_accuracy']:.3f}" if r["dev_accuracy"] is not None else "  —  "
        train = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "  —  "
        logger.info(
            "  %4d  %8s  %10s  %8s  %9s  %10d  %8d",
            r["iteration"],
            r["id"],
            r["cot_format"],
            dev,
            train,
            r["meta_tokens"],
            r["qwen_tokens"],
        )
