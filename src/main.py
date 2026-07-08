"""
main.py — CLI entry point for EvoAgent.

Usage:
    python main.py \
        --T 5 \
        --output-dir ./runs/experiment_01 \
        --train-size 100 \
        --dev-size 200

Run `python main.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Suppress harmless loky/multiprocessing resource_tracker noise at process exit.
warnings.filterwarnings("ignore", message="resource_tracker")


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure root logger to write to both stderr and a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Console handler.
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler.
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.info("Logging to %s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoAgent: self-improving LLM agent for Vietnamese reading comprehension.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core loop parameters
    parser.add_argument(
        "--T",
        type=int,
        default=5,
        help="Number of EvoAgent iterations (including the seed at iteration 0).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to an existing history.jsonl file to resume from.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="local_financial_qa",
        help="Name of the dataset (e.g. local_financial_qa).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Number of training examples to use for the cheap mid-loop eval.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=None,
        help="Number of dev examples (None = use full dev split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="QuantTrio/Qwen3.5-4B-AWQ",
        help="HuggingFace model ID or local path for inference.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per completion.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length for SGLang.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 = greedy (recommended for evaluation).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit quantization (requires more VRAM; use for testing on GPU with >=24 GB).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use for SGLang.",
    )

    # Always-From-Original (AFO) Principle
    parser.add_argument(
        "--afo-mode",
        type=str,
        default="best",
        choices=["none", "best", "original", "probabilistic"],
        help="Parent selection mode: none (always latest), best (always best), original (always seed), probabilistic (mix of best/original/latest).",
    )
    parser.add_argument(
        "--afo-prob-best",
        type=float,
        default=0.4,
        help="Probability weight to select the best strategy as parent in probabilistic AFO mode.",
    )
    parser.add_argument(
        "--afo-prob-original",
        type=float,
        default=0.3,
        help="Probability weight to select the original seed strategy as parent in probabilistic AFO mode.",
    )
    parser.add_argument(
        "--afo-prob-latest",
        type=float,
        default=0.3,
        help="Probability weight to select the latest strategy as parent in probabilistic AFO mode.",
    )
    parser.add_argument(
        "--progressive-reflections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable progressive context management in reflections (reduce failures shown over time).",
    )
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        default=False,
        help="Enable curriculum learning (order training examples by difficulty).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/default",
        help="Directory to save history, eval results, and analysis plots.",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=1.0,
        help="Stop early if dev accuracy reaches this threshold.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip post-run analysis plots (useful for quick tests).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run Stage 4 pre-flight smoke test on the seed strategy instead of the full optimization loop.",
    )

    return parser.parse_args()


def load_dataset_splits(
    dataset_id: str,
    train_size: int,
    dev_size: int | None,
    seed: int,
):
    """
    Load data splits from local JSON files.
    """
    from src.data import load_data_splits
    return load_data_splits(train_size=train_size, dev_size=dev_size, seed=seed)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("EvoAgent starting. Args: %s", vars(args))

    # Save args to output dir for reproducibility.
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    train_subset, dev_split = load_dataset_splits(
        dataset_id=args.dataset,
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
    )

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    from src.model import QwenInference

    logger.info("Initialising QwenInference with model '%s'…", args.model)
    model = QwenInference(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_4bit=not args.no_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    model.load()

    # ----------------------------------------------------------------
    # Run EvoAgent / Smoke Test
    # ----------------------------------------------------------------
    from src.harness import run_smoke_test, run_evoagent
    from src.strategy import make_seed_strategy

    if args.smoke_test:
        logger.info("Running pre-flight smoke test on seed strategy...")
        seed_strategy = make_seed_strategy()
        passed = run_smoke_test(seed_strategy, train_subset, model)
        if passed:
            logger.info("Smoke test PASSED!")
            print("\nSMOKE TEST PASSED!\n")
            sys.exit(0)
        else:
            logger.error("Smoke test FAILED!")
            print("\nSMOKE TEST FAILED!\n")
            sys.exit(1)

    history = run_evoagent(
        T=args.T,
        train_dataset=train_subset,
        dev_dataset=dev_split,
        train_size=args.train_size,
        model=model,
        output_dir=output_dir,
        resume_from=Path(args.resume) if args.resume else None,
        early_stop_accuracy=args.early_stop,
        afo_mode=args.afo_mode,
        afo_prob_best=args.afo_prob_best,
        afo_prob_original=args.afo_prob_original,
        afo_prob_latest=args.afo_prob_latest,
        progressive_reflections=args.progressive_reflections,
        use_curriculum=args.use_curriculum,
    )

    # ----------------------------------------------------------------
    # Post-run analysis
    # ----------------------------------------------------------------
    if not args.skip_analysis:
        from src.analysis import (
            compute_strategy_diversity,
            failure_mode_report,
            plot_learning_curve,
        )

        logger.info("Running post-run analysis…")
        try:
            plot_learning_curve(history, output_dir=output_dir)
        except Exception as exc:
            logger.error("Learning curve plot failed: %s", exc)

        try:
            compute_strategy_diversity(history, output_dir=output_dir)
        except Exception as exc:
            logger.error("Diversity computation failed: %s", exc)

        try:
            report_text, _ = failure_mode_report(history, output_dir=output_dir)
            print("\n" + report_text)
        except Exception as exc:
            logger.error("Failure mode report failed: %s", exc)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    best = history.best_strategy()
    if best:
        logger.info(
            "Run complete. Best strategy: iteration=%d, dev_accuracy=%.3f.",
            best.metadata.iteration,
            best.metadata.dev_accuracy,
        )
        print(
            f"\nEvoAgent complete.\n"
            f"Best strategy: iteration {best.metadata.iteration}, "
            f"dev accuracy = {best.metadata.dev_accuracy:.3f}\n"
            f"Results saved to: {output_dir.resolve()}"
        )
    else:
        logger.info("Run complete. No strategies were evaluated.")


if __name__ == "__main__":
    main()
