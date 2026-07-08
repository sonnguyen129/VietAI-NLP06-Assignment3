"""
submit.py — Generate test set predictions for Kaggle submission.

Usage:
    python submit.py \
        --strategy-path ./runs/exp_self/iter_003_strategy.json \
        --output-file ./submission.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure the root of the workspace is in the import path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from src.data import load_dataset
from src.executor import build_prompt
from src.evaluator import evaluate_program
from src.model import QwenInference, extract_answer
from src.strategy import Strategy, CoTFormat
from format_submission import clean_prediction_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoAgent: Generate Kaggle test set submission using the best evolved strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy-path",
        type=str,
        required=True,
        help="Path to the strategy JSON file to use for generating prompts.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="submission.csv",
        help="Path where the output submission.csv will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test questions to process (useful for quick checks).",
    )
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
        "--no-4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit quantization.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use for SGLang.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_path = Path(args.strategy_path)
    output_file = Path(args.output_file)

    if not strategy_path.exists():
        logger.error("Strategy file not found: %s", strategy_path)
        sys.exit(1)

    logger.info("Loading strategy from %s...", strategy_path)
    try:
        strategy = Strategy.from_json(strategy_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load strategy: %s", e)
        sys.exit(1)

    logger.info("Loading test dataset splits...")
    try:
        ds = load_dataset()
        test_dataset = ds["test"]
        if args.limit is not None:
            logger.info("Limiting test dataset to first %d examples.", args.limit)
            test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    logger.info("Initializing QwenInference with model '%s'...", args.model)
    model = QwenInference(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,  # Greedy for determinism
        use_4bit=not args.no_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    model.load()

    logger.info("Generating prompts for %d test questions...", len(test_dataset))
    from tqdm import tqdm
    prompts = []
    for row in tqdm(test_dataset, desc="Generating prompts"):
        user_message = build_prompt(strategy, row["context"], row["question"])
        formatted_prompt = model.format_prompt(
            system_message=(
                "Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. "
                "Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi dựa trên văn bản và bảng số liệu được cung cấp."
            ),
            user_message=user_message,
            enable_thinking=(strategy.cot_format != CoTFormat.NONE),
        )
        prompts.append(formatted_prompt)

    # TODO: Set appropriate max generation bounds
    model.max_new_tokens = 4096 if strategy.cot_format != CoTFormat.NONE else 256

    logger.info("Running batch inference on test set...")
    with tqdm(total=len(prompts), desc="Running inference") as pbar:
        raw_outputs = model.generate_batch(
            prompts,
            cot_format=(strategy.cot_format != CoTFormat.NONE)
        )
        pbar.update(len(raw_outputs))

    logger.info("Parsing programs and generating submission file at %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    debug_file = output_file.parent / f"{output_file.stem}_details.json"
    debug_records = []

    with output_file.open(mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Usage", "predicted_value"])

        for row, raw_out in tqdm(zip(test_dataset, raw_outputs), total=len(test_dataset), desc="Saving predictions"):
            pred_program = raw_out.predicted_answer
            
            # Execute the program to get executed value
            try:
                pred_val = evaluate_program(pred_program, row["table"])
            except Exception:
                pred_val = None
            pred_val = clean_prediction_value(pred_val)

            usage = "Public"
            writer.writerow([row["id"], usage, pred_val])
            
            debug_records.append({
                "id": row["id"],
                "question": row["question"],
                "raw_output": raw_out.raw_output,
                "program": pred_program,
                "predicted_value": pred_val
            })

    logger.info("Saving debug details to %s...", debug_file)
    with debug_file.open(mode="w", encoding="utf-8") as df:
        json.dump(debug_records, df, ensure_ascii=False, indent=2)

    logger.info("Test set submission successfully generated! Total predictions: %d", len(test_dataset))


if __name__ == "__main__":
    main()
