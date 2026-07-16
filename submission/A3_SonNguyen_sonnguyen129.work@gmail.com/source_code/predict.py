"""
predict.py — Phase-3 (Kaggle leaderboard) pipeline CLI.

Subcommands:
    rescore   Re-run extraction v2 + evaluator v2 over EXISTING raw outputs
              (no GPU) to measure the offline lift before spending GPU hours.
    generate  Generate candidate programs (greedy/sampled groups) with the
              robust retry+cache pipeline. Resumable.
    repair    Guided-JSON repair pass for samples without an executable program.
    score     Score an ablation config on the validation split (gold available).
    judge     (OPTIONAL) LLM-judge low-consensus questions (requires the
              Cerebrium app to be swapped to the judge model).
    submit    Aggregate final per-question values -> test_predictions.json
              (feed that to format_submission.py to produce submission.csv).
    check     Sanity-gate a submission.csv before uploading to Kaggle.

All runs append to runs/phase3/run.log and snapshot their parameters into
runs/phase3/phase3_config.json (Phase-1-style artifacts).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

PHASE3_DIR = ROOT / "runs" / "phase3"

logger = logging.getLogger("phase3")

ABS_TOL = 1e-4


# ------------------------------------------------------------------
# Infra: logging + config snapshot (Phase-1-style artifacts)
# ------------------------------------------------------------------

def setup_logging(subcommand: str) -> None:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        f"%(asctime)s [%(levelname)s] [{subcommand}] %(name)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file_handler = logging.FileHandler(PHASE3_DIR / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)


def snapshot_config(subcommand: str, args: argparse.Namespace) -> None:
    path = PHASE3_DIR / "phase3_config.json"
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subcommand": subcommand,
        "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
    }
    history.append(entry)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def load_split(split: str):
    """Load a dataset split ('validation' | 'test' | 'train') as a list of rows."""
    from src.data import load_dataset

    ds = load_dataset()
    return list(ds[split])


def gold_value(row_or_record: dict) -> float | None:
    """Gold numeric value from exe_ans (fallback: execute the gold program)."""
    exe_ans = row_or_record.get("exe_ans")
    if exe_ans not in (None, ""):
        try:
            return float(str(exe_ans).replace(",", "").replace("%", ""))
        except ValueError:
            pass
    gv = row_or_record.get("gold_val")
    if gv is not None:
        try:
            return float(gv)
        except (TypeError, ValueError):
            pass
    gold_prog = row_or_record.get("answer") or row_or_record.get("gold_answer")
    table = row_or_record.get("table") or []
    if gold_prog:
        try:
            from src.evaluator import evaluate_program

            return evaluate_program(gold_prog, table)
        except Exception:
            return None
    return None


def is_correct_value(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= ABS_TOL


# ------------------------------------------------------------------
# rescore — offline lift check on existing raw outputs (P1 gate)
# ------------------------------------------------------------------

def cmd_rescore(args: argparse.Namespace) -> None:
    from src.evaluator import evaluate_program_v2
    from src.extraction_v2 import extract_program_v2

    source = Path(args.source)
    data = json.loads(source.read_text(encoding="utf-8"))
    records = data["per_question"] if isinstance(data, dict) else data
    logger.info("Rescoring %d records from %s", len(records), source)

    # Join tables back from dev.json by (question, gold program).
    dev_rows = load_split("validation")
    table_map = {}
    for row in dev_rows:
        table_map[(row["question"].strip(), (row.get("answer") or "").strip())] = row

    old_correct = 0
    new_correct = 0
    tag_counts: dict[str, int] = {}
    rescued, lost = [], []
    per_question_out = []

    for rec in records:
        gold_prog = (rec.get("gold_answer") or "").strip()
        key = (rec.get("question", "").strip(), gold_prog)
        dev_row = table_map.get(key)
        table = dev_row["table"] if dev_row else []
        gold = gold_value({**rec, "exe_ans": dev_row.get("exe_ans") if dev_row else None,
                           "answer": gold_prog, "table": table})

        old_ok = bool(rec.get("is_correct"))
        old_correct += int(old_ok)

        extraction = extract_program_v2(rec.get("raw_output") or "")
        tag_counts[extraction.tag] = tag_counts.get(extraction.tag, 0) + 1

        new_val = None
        if extraction.program:
            try:
                new_val = evaluate_program_v2(extraction.program, table)
            except Exception:
                new_val = None
        new_ok = is_correct_value(new_val, gold)
        new_correct += int(new_ok)

        if new_ok and not old_ok:
            rescued.append(rec.get("question_id"))
        elif old_ok and not new_ok:
            lost.append({
                "question_id": rec.get("question_id"),
                "old_program": rec.get("predicted_answer"),
                "new_program": extraction.program,
                "tag": extraction.tag,
            })

        per_question_out.append({
            "question_id": rec.get("question_id"),
            "question": rec.get("question"),
            "gold_answer": gold_prog,
            "gold_val": gold,
            "predicted_answer": extraction.program,
            "predicted_val": new_val,
            "is_correct": new_ok,
            "question_type": rec.get("question_type"),
            "extraction_tag": extraction.tag,
            "old_is_correct": old_ok,
        })

    total = len(records)
    old_acc = old_correct / total if total else 0.0
    new_acc = new_correct / total if total else 0.0
    report = {
        "source": str(source),
        "total": total,
        "old_accuracy": round(old_acc, 5),
        "new_accuracy": round(new_acc, 5),
        "lift_pp": round((new_acc - old_acc) * 100, 2),
        "extraction_tags": tag_counts,
        "rescued": len(rescued),
        "lost": len(lost),
        "lost_details": lost,
        "note": "Rescore is offline-only: runaway/empty outputs stay wrong here — the repair pass (GPU) handles them later.",
    }
    out_path = PHASE3_DIR / "rescore_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_path = PHASE3_DIR / "rescore_eval.json"
    eval_path.write_text(
        json.dumps({"per_question": per_question_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("OLD accuracy: %.4f (%d/%d)", old_acc, old_correct, total)
    logger.info("NEW accuracy: %.4f (%d/%d)  lift=%+.2fpp", new_acc, new_correct, total, (new_acc - old_acc) * 100)
    logger.info("Extraction tags: %s", tag_counts)
    logger.info("Rescued %d, lost %d. Report -> %s", len(rescued), len(lost), out_path)
    if (new_acc - old_acc) * 100 < 3:
        logger.warning("GATE P1 CHƯA ĐẠT (+3pp) — xem lost_details trong %s", out_path)
    else:
        logger.info("GATE P1 ĐẠT (>= +3pp).")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rescore", help="Offline lift check on existing raw outputs")
    p.add_argument("--source", default=str(ROOT / "runs" / "exp_self" / "iter_004_eval_dev.json"))
    p.set_defaults(func=cmd_rescore)

    # generate / repair / score / judge / submit / check are registered by
    # pipeline_cli (kept separate so `rescore` works without pipeline deps).
    try:
        from src.pipeline_v2 import register_cli

        register_cli(sub)
    except ImportError:
        pass

    args = parser.parse_args()
    setup_logging(args.command)
    snapshot_config(args.command, args)
    args.func(args)


if __name__ == "__main__":
    main()
