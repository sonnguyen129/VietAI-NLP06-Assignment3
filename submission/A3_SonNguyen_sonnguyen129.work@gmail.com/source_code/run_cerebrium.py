"""
run_cerebrium.py — Local runners for the EvoAgent milestone jobs.

Replaces run_modal.py. The evolution loop, graders, and proof generation all
run locally; only model inference goes over HTTP to a vLLM OpenAI-compatible
server deployed on Cerebrium (see cerebrium.toml).

Prerequisites:
  1. cerebrium deploy                       (once, from this folder)
  2. Fill .env with CEREBRIUM_BASE_URL / CEREBRIUM_API_KEY / HF_TOKEN

Usage:
  python run_cerebrium.py sandbox     # Stage 0 -> sandbox_proof.json
  python run_cerebrium.py smoke       # Stage 4 pre-flight -> smoke_proof.json
  python run_cerebrium.py test        # small confidence run (train/dev 32)
  python run_cerebrium.py main        # full proof run -> evolution_proof.json
  python run_cerebrium.py submit --strategy-path runs/exp_self/iter_XXX_strategy.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"


def run_sandbox(args: argparse.Namespace) -> None:
    """Stage 0: zero-shot baseline on 50 dev examples -> sandbox_proof.json."""
    from src.sandbox import get_model, run_sandbox_accuracy_check

    model = get_model(args.model)
    eval_res = run_sandbox_accuracy_check(model, dev_size=50)

    result = {
        "status": "success",
        "baseline_accuracy": eval_res.get("accuracy", 0.0),
        "num_correct": eval_res.get("num_correct", 0),
        "num_examples": eval_res.get("num_examples", 0),
        "message": (
            f"Sandbox ran successfully via Cerebrium. Baseline Accuracy: "
            f"{eval_res.get('accuracy', 0.0) * 100:.2f}% "
            f"({eval_res.get('num_correct')}/{eval_res.get('num_examples')})"
        ),
        "samples": eval_res.get("samples", []),
    }

    proof_path = ROOT / "sandbox_proof.json"
    proof_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {proof_path}")


def run_smoke(args: argparse.Namespace) -> None:
    """Stage 4 pre-flight smoke test on the seed strategy -> smoke_proof.json."""
    proc = subprocess.run(
        [
            sys.executable, "main.py",
            "--smoke-test",
            "--dataset", "local_financial_qa",
            "--model", args.model,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    print(proc.stdout[-2000:])
    passed = proc.returncode == 0
    result = {
        "status": "success" if passed else "failed",
        "smoke_test_passed": passed,
        "returncode": proc.returncode,
        "message": "Smoke test passed." if passed else "Smoke test failed.",
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }
    proof_path = ROOT / "smoke_proof.json"
    proof_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {proof_path}")
    if not passed:
        raise SystemExit(1)


def run_test(args: argparse.Namespace) -> None:
    """Small confidence run to catch integration issues before the full run."""
    output_dir = ROOT / "runs" / "exp_test"
    shutil.rmtree(output_dir, ignore_errors=True)
    subprocess.run(
        [
            sys.executable, "main.py",
            "--T", "5",
            "--dataset", "local_financial_qa",
            "--output-dir", str(output_dir),
            "--train-size", "32",
            "--dev-size", "32",
            "--model", args.model,
            "--skip-analysis",
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Test run finished. Results in {output_dir}")


def run_main(args: argparse.Namespace) -> None:
    """Full self-evolution proof run -> evolution_proof.json."""
    output_dir = ROOT / "runs" / "exp_self"
    if not args.resume:
        shutil.rmtree(output_dir, ignore_errors=True)
    subprocess.run(
        [
            sys.executable, "main.py",
            "--T", "5",
            "--dataset", "local_financial_qa",
            "--output-dir", str(output_dir),
            "--train-size", "200",
            "--dev-size", "240",
            "--model", args.model,
            "--progressive-reflections",
            "--use-curriculum",
            "--afo-mode", "best",
        ],
        cwd=ROOT,
        check=True,
    )

    from src.strategy import StrategyHistory

    history_path = output_dir / "history.jsonl"
    if not history_path.exists():
        print("Error: history.jsonl not found. Cannot generate evolution proof.")
        raise SystemExit(1)

    history = StrategyHistory(history_path)
    history.load()
    best_strategy = history.best_strategy()
    best_acc = best_strategy.metadata.dev_accuracy if best_strategy else 0.0

    proof_data = {
        "status": "success",
        "best_iteration": best_strategy.metadata.iteration if best_strategy else None,
        "best_dev_accuracy": best_acc,
        "baseline_accuracy": 0.42,
        "history": [
            {
                "iteration": s.metadata.iteration,
                "dev_accuracy": s.metadata.dev_accuracy,
            }
            for s in history.strategies
        ],
    }
    for proof_path in (output_dir / "evolution_proof.json", ROOT / "evolution_proof.json"):
        proof_path.write_text(
            json.dumps(proof_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {proof_path}")
    print(f"Best dev accuracy: {best_acc}")


def run_submit(args: argparse.Namespace) -> None:
    """Generate the Kaggle test submission from a strategy."""
    cmd = [
        sys.executable, "submit.py",
        "--strategy-path", args.strategy_path,
        "--output-file", args.output_file,
        "--model", args.model,
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Submission saved to {args.output_file}")


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id served by the Cerebrium vLLM endpoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sandbox", help="Stage 0 baseline -> sandbox_proof.json")
    sub.add_parser("smoke", help="Stage 4 smoke test -> smoke_proof.json")
    sub.add_parser("test", help="Small confidence evolution run (train/dev 32)")

    p_main = sub.add_parser("main", help="Full proof run -> evolution_proof.json")
    p_main.add_argument("--resume", action="store_true", help="Keep existing runs/exp_self and resume from its history.")

    p_submit = sub.add_parser("submit", help="Kaggle submission from a strategy JSON")
    p_submit.add_argument("--strategy-path", required=True)
    p_submit.add_argument("--output-file", default="submission.csv")
    p_submit.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    {
        "sandbox": run_sandbox,
        "smoke": run_smoke,
        "test": run_test,
        "main": run_main,
        "submit": run_submit,
    }[args.command](args)


if __name__ == "__main__":
    main()
