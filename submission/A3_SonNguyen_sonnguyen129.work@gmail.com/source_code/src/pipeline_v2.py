"""
pipeline_v2.py — Phase-3 generation pipeline: JSONL cache/resume, retrying
scheduler, repair pass, ablation scoring, submission aggregation, sanity gate.

Registered into predict.py via register_cli(). All artifacts live under
runs/phase3/ (Phase-1-style logging & evidence).
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
PHASE3_DIR = ROOT / "runs" / "phase3"

ABS_TOL = 1e-4
DEFAULT_MODEL = os.environ.get("CEREBRIUM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")

GROUP_SPECS = {
    "greedy": dict(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                   repetition_penalty=1.05, max_new_tokens=3072),
    "sampled": dict(temperature=0.6, top_p=0.95, top_k=20,
                    repetition_penalty=1.05, max_new_tokens=2048),
}


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

@dataclass
class GenRecord:
    qid: str
    strategy_id: str
    sample_index: int          # 0 = greedy; 1..K = sampled
    status: str                # ok | failed | repaired | repair_failed
    raw_output: str
    program: Optional[str]
    value: Optional[float]
    stated_result: Optional[float]
    extraction_tag: str
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    ts: float


class PredictionCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[tuple[str, str, int], GenRecord] = {}
        self.load()

    def load(self) -> None:
        self.records.clear()
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = GenRecord(**d)
                except Exception:
                    continue  # tolerate a truncated last line
                self.records[(rec.qid, rec.strategy_id, rec.sample_index)] = rec

    def append(self, rec: GenRecord) -> None:
        self.records[(rec.qid, rec.strategy_id, rec.sample_index)] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def get(self, qid: str, strategy_id: str, sample_index: int) -> Optional[GenRecord]:
        return self.records.get((qid, strategy_id, sample_index))


# ------------------------------------------------------------------
# Strategy / prompt helpers
# ------------------------------------------------------------------

def load_base_strategy(name: str):
    from src.strategy import Strategy

    path = PHASE3_DIR / f"strategy_{name}.json"
    if name == "retr":
        path = PHASE3_DIR / "strategy_s4fix.json"  # retr shares the s4fix base
    return Strategy.from_json(path.read_text(encoding="utf-8"))


def build_row_prompt(model, strategy, row: dict) -> str:
    from src.executor import _SYSTEM_MESSAGE, build_prompt
    from src.strategy import CoTFormat

    user_message = build_prompt(strategy, row["context"], row["question"])
    return model.format_prompt(
        system_message=_SYSTEM_MESSAGE,
        user_message=user_message,
        enable_thinking=strategy.cot_format != CoTFormat.NONE,
    )


def make_model():
    from src.model import QwenInference

    model = QwenInference(model_name_or_path=DEFAULT_MODEL, max_model_len=16384)
    model.load()
    return model


def process_choice(gen, row: dict) -> dict:
    """Extract + execute one generation choice -> fields for GenRecord."""
    from src.evaluator import evaluate_program_v2
    from src.extraction_v2 import extract_program_v2

    extraction = extract_program_v2(gen.raw_output)
    value = None
    error = None
    if extraction.program:
        try:
            value = evaluate_program_v2(extraction.program, row.get("table") or [])
        except Exception as exc:
            error = f"exec: {exc}"
    return dict(
        raw_output=gen.raw_output,
        program=extraction.program,
        value=value,
        stated_result=extraction.stated_result,
        extraction_tag=extraction.tag,
        error=error,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
    )


def _update_token_usage(new_input: int, new_output: int, bucket: str) -> None:
    path = PHASE3_DIR / "token_usage.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    entry = data.setdefault(bucket, {"input_tokens": 0, "output_tokens": 0})
    entry["input_tokens"] += new_input
    entry["output_tokens"] += new_output
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------
# generate
# ------------------------------------------------------------------

def cmd_generate(args) -> None:
    from predict import load_split

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    cache = PredictionCache(Path(args.cache))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    k = args.k

    model = make_model()
    model.assert_served_model(DEFAULT_MODEL)

    retrieval_index = None
    if "retr" in strategies:
        from src.retrieval import TrainIndex

        retrieval_index = TrainIndex.build_default()

    base_strategies = {name: load_base_strategy(name) for name in strategies}

    # Build the task list, skipping request-groups fully satisfied by the cache.
    # A group is "done" when every sample index has an ok/repaired record.
    tasks = []  # (row, strategy_name, group_name, sample_indices)
    for row in rows:
        for strat_name in strategies:
            for group in groups:
                indices = [0] if group == "greedy" else list(range(1, k + 1))
                done = all(
                    (rec := cache.get(row["id"], strat_name, i)) is not None
                    and rec.status in ("ok", "repaired")
                    for i in indices
                )
                if not done:
                    tasks.append((row, strat_name, group, indices))

    logger.info(
        "generate: split=%s rows=%d strategies=%s groups=%s k=%d -> %d pending request-groups",
        args.split, len(rows), strategies, groups, k, len(tasks),
    )
    if not tasks:
        logger.info("Nothing to do — cache complete.")
        return

    def run_group(task):
        row, strat_name, group, indices = task
        if strat_name == "retr":
            from src.retrieval import strategy_for_row

            strategy = strategy_for_row(base_strategies["retr"], row, retrieval_index)
        else:
            strategy = base_strategies[strat_name]
        prompt = build_row_prompt(model, strategy, row)
        spec = dict(GROUP_SPECS[group])
        if group == "sampled":
            import zlib

            spec["n"] = len(indices)
            spec["seed"] = zlib.crc32(row["id"].encode("utf-8")) % (2 ** 31)
        try:
            gens = model.complete_many(prompt, **spec)
            return task, gens, None
        except Exception as exc:
            return task, None, exc

    from tqdm import tqdm

    total_in = total_out = ok_count = fail_count = runaway_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_group, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
            (row, strat_name, group, indices), gens, exc = fut.result()
            now = time.time()
            if exc is not None:
                logger.error("group failed qid=%s strat=%s group=%s: %s", row["id"], strat_name, group, exc)
                for i in indices:
                    cache.append(GenRecord(row["id"], strat_name, i, "failed", "", None, None,
                                           None, "", f"{type(exc).__name__}: {exc}", 0, 0, now))
                fail_count += len(indices)
                continue
            for i, gen in zip(indices, gens):
                fields = process_choice(gen, row)
                rec = GenRecord(row["id"], strat_name, i, "ok", ts=now, **fields)
                cache.append(rec)
                total_in += rec.input_tokens
                total_out += rec.output_tokens
                ok_count += 1
                if rec.extraction_tag == "runaway":
                    runaway_count += 1

    _update_token_usage(total_in, total_out, "qwen_generate")
    logger.info(
        "generate done: ok=%d failed=%d runaway=%d (%.1f%%) tokens in=%d out=%d",
        ok_count, fail_count, runaway_count,
        100 * runaway_count / max(1, ok_count), total_in, total_out,
    )


# ------------------------------------------------------------------
# repair
# ------------------------------------------------------------------

def _needs_repair(rec: GenRecord) -> bool:
    if rec.status == "failed":
        return True
    if rec.status in ("ok",) and (rec.program is None or rec.value is None):
        return True
    return False


def cmd_repair(args) -> None:
    from predict import load_split
    from src.evaluator import evaluate_program_v2
    from src.extraction_v2 import (REPAIR_SCHEMA, build_repair_message,
                                   repair_system_message, extract_program_v2)

    rows = {row["id"]: row for row in load_split(args.split)}
    cache = PredictionCache(Path(args.cache))
    model = make_model()
    model.assert_served_model(DEFAULT_MODEL)

    targets = [rec for rec in cache.records.values()
               if rec.qid in rows and _needs_repair(rec)]
    logger.info("repair: %d samples need repair", len(targets))
    if not targets:
        return

    def run_repair(rec: GenRecord):
        row = rows[rec.qid]
        message = build_repair_message(row["question"], row["context"], rec.raw_output or "")
        prompt = model.format_prompt(
            system_message=repair_system_message(),
            user_message=message,
            enable_thinking=False,
        )
        try:
            gens = model.complete_many(
                prompt, n=1, temperature=0.0, top_p=1.0, top_k=-1,
                repetition_penalty=1.0, max_new_tokens=256,
                guided_json=REPAIR_SCHEMA, max_attempts=4,
            )
            return rec, gens[0], None
        except Exception as exc:
            return rec, None, exc

    from tqdm import tqdm

    repaired = failed = 0
    total_in = total_out = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_repair, r) for r in targets]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="repair"):
            rec, gen, exc = fut.result()
            now = time.time()
            row = rows[rec.qid]
            if exc is not None or gen is None:
                cache.append(GenRecord(rec.qid, rec.strategy_id, rec.sample_index,
                                       "repair_failed", rec.raw_output, rec.program, rec.value,
                                       rec.stated_result, rec.extraction_tag,
                                       f"repair: {exc}", rec.input_tokens, rec.output_tokens, now))
                failed += 1
                continue
            total_in += gen.input_tokens
            total_out += gen.output_tokens
            extraction = extract_program_v2("</think>" + (gen.raw_output or ""))
            program = extraction.program
            if program is None:
                # guided JSON returns a bare JSON object without think tags
                try:
                    payload = json.loads(gen.raw_output, strict=False)
                    from src.extraction_v2 import _validate_program

                    program = _validate_program(payload.get("Program syntax", ""))
                except Exception:
                    program = None
            value = None
            error = None
            if program:
                try:
                    value = evaluate_program_v2(program, row.get("table") or [])
                except Exception as e2:
                    error = f"exec: {e2}"
            status = "repaired" if value is not None else "repair_failed"
            cache.append(GenRecord(rec.qid, rec.strategy_id, rec.sample_index, status,
                                   rec.raw_output, program, value, extraction.stated_result,
                                   "repair", error, rec.input_tokens + gen.input_tokens,
                                   rec.output_tokens + gen.output_tokens, now))
            repaired += int(status == "repaired")
            failed += int(status != "repaired")

    _update_token_usage(total_in, total_out, "qwen_repair")
    logger.info("repair done: repaired=%d still-failed=%d", repaired, failed)


# ------------------------------------------------------------------
# Candidate assembly + ablation configs
# ------------------------------------------------------------------

ABLATIONS = {
    "A1": dict(strategies=["s4fix"], greedy_only=True, judge=False),
    "A2": dict(strategies=["s4fix"], greedy_only=True, judge=False),
    "A3": dict(strategies=["s4fix"], greedy_only=False, judge=False),
    "A4": dict(strategies=["s4fix", "retr"], greedy_only=False, judge=False),
    "A5": dict(strategies=["s4fix", "retr"], greedy_only=False, judge=True),
}


def assemble_candidates(cache: PredictionCache, qid: str, strategies: list[str],
                        greedy_only: bool):
    from src.voting import Candidate

    cands = []
    for (rqid, strat, idx), rec in cache.records.items():
        if rqid != qid or strat not in strategies:
            continue
        if greedy_only and idx != 0:
            continue
        if rec.status not in ("ok", "repaired") or rec.value is None:
            continue
        cands.append(Candidate(value=rec.value, program=rec.program,
                               strategy_id=strat, sample_index=idx,
                               stated_result=rec.stated_result))
    return cands


def load_judge_verdicts(split: str) -> dict:
    path = PHASE3_DIR / "judge_cache.jsonl"
    verdicts = {}
    if not path.exists():
        return verdicts
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("split") == split and d.get("final_value") is not None:
                verdicts[d["qid"]] = d
    return verdicts


def resolve_question(cache: PredictionCache, qid: str, config: dict,
                     judge_verdicts: dict, vote_cfg=None):
    """Return (final_value, meta) for one question under an ablation config."""
    from src.voting import cluster_and_vote

    cands = assemble_candidates(cache, qid, config["strategies"], config["greedy_only"])
    outcome = cluster_and_vote(cands, vote_cfg)
    final_value = outcome.value
    source = "vote"
    if config.get("judge") and outcome.needs_judge and qid in judge_verdicts:
        final_value = judge_verdicts[qid]["final_value"]
        source = "judge"
    meta = dict(
        num_candidates=len(cands),
        confidence=round(outcome.confidence, 3),
        margin=round(outcome.margin, 3),
        needs_judge=outcome.needs_judge,
        reason=outcome.reason,
        source=source,
        clusters=[{"value": c.rep_value, "weight": round(c.weight, 2),
                   "n": len(c.members),
                   "program": c.members[0].program} for c in outcome.clusters[:3]],
    )
    return final_value, meta


# ------------------------------------------------------------------
# score (validation only)
# ------------------------------------------------------------------

def cmd_score(args) -> None:
    from predict import gold_value, is_correct_value, load_split
    from src.executor import classify_question_type

    config = ABLATIONS[args.ablation]
    rows = load_split("validation")
    if args.limit:
        rows = rows[: args.limit]
    cache = PredictionCache(Path(args.cache))
    judge_verdicts = load_judge_verdicts("validation") if config.get("judge") else {}

    per_question = []
    correct = 0
    by_type: dict[str, list[int]] = {}
    judged = 0
    no_cands = 0
    for row in rows:
        gold = gold_value(row)
        value, meta = resolve_question(cache, row["id"], config, judge_verdicts)
        ok = is_correct_value(value, gold)
        correct += int(ok)
        judged += int(meta["source"] == "judge")
        no_cands += int(meta["reason"] == "no_candidates")
        q_type = classify_question_type(row.get("answer") or "")
        by_type.setdefault(q_type, []).append(int(ok))
        per_question.append(dict(
            question_id=row["id"], question=row["question"],
            gold_answer=row.get("answer"), gold_val=gold,
            predicted_answer=(meta["clusters"][0]["program"] if meta["clusters"] else None),
            predicted_val=value, is_correct=ok,
            question_type=q_type, vote=meta,
        ))

    total = len(rows)
    accuracy = correct / total if total else 0.0
    acc_by_type = {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())}
    logger.info("[%s] accuracy = %.4f (%d/%d) | judge-used=%d | no-candidates=%d",
                args.ablation, accuracy, correct, total, judged, no_cands)
    logger.info("[%s] by type: %s", args.ablation, acc_by_type)

    eval_path = PHASE3_DIR / f"ablation_{args.ablation}_eval.json"
    eval_path.write_text(json.dumps({
        "ablation": args.ablation, "accuracy": accuracy, "num_examples": total,
        "num_correct": correct, "accuracy_by_type": acc_by_type,
        "per_question": per_question,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = PHASE3_DIR / "ablation_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report[args.ablation] = {
        "accuracy": round(accuracy, 5), "num_correct": correct, "total": total,
        "accuracy_by_type": acc_by_type, "judge_used": judged,
        "no_candidates": no_cands, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %s and updated %s", eval_path.name, report_path.name)


# ------------------------------------------------------------------
# submit (test aggregation) + check
# ------------------------------------------------------------------

def cmd_submit(args) -> None:
    from predict import load_split

    config = ABLATIONS[args.ablation]
    rows = load_split("test")
    cache = PredictionCache(Path(args.cache))
    judge_verdicts = load_judge_verdicts("test") if config.get("judge") else {}

    predictions = {}
    details = []
    missing = 0
    for row in rows:
        value, meta = resolve_question(cache, row["id"], config, judge_verdicts)
        if value is None:
            missing += 1
            value = 0.0
        predictions[row["id"]] = value
        details.append(dict(qid=row["id"], question=row["question"],
                            final_value=value, **meta))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path = PHASE3_DIR / "test_details.json"
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("submit: %d predictions (%d without candidates -> 0.0) -> %s",
                len(predictions), missing, out)
    logger.info("Next: python format_submission.py --predictions %s --output-file submission.csv", out)


def _plain_decimal(value: float) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def cmd_check(args) -> None:
    import csv

    from predict import load_split

    sub_path = Path(args.submission)
    rows = load_split("test")
    expected_ids = [str(r["id"]) for r in rows]

    with sub_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data_rows = list(reader)

    problems = []
    if header != ["id", "Usage", "predicted_value"]:
        problems.append(f"Header sai: {header}")
    ids = [r[0] for r in data_rows]
    if ids != expected_ids:
        problems.append(f"IDs không khớp/không đúng thứ tự (rows={len(ids)}, expected={len(expected_ids)})")
    values = []
    e_notation_rows = []
    for i, r in enumerate(data_rows):
        if len(r) != 3:
            problems.append(f"Dòng {i+2}: số cột != 3")
            continue
        raw_val = r[2]
        if "e" in raw_val.lower() or raw_val in ("nan", "inf", "-inf", "") or raw_val == "-0.0":
            e_notation_rows.append((i, raw_val))
        try:
            values.append(float(raw_val))
        except ValueError:
            problems.append(f"Dòng {i+2}: giá trị không parse được: {raw_val!r}")

    zeros = sum(1 for v in values if v == 0.0)
    negatives = sum(1 for v in values if v < 0)
    ratio_like = sum(1 for v in values if abs(v) <= 1.5)
    logger.info("check: rows=%d zeros=%d negatives=%d ratio-like=%d e-notation=%d",
                len(values), zeros, negatives, ratio_like, len(e_notation_rows))
    if zeros > 10:
        problems.append(f"Có {zeros} giá trị 0.0 (>10) — nghi silent failure (baseline từng có 102)")

    if e_notation_rows and args.fix:
        logger.info("Fixup render số cho %d ô (giá trị giữ nguyên)...", len(e_notation_rows))
        for i, _ in e_notation_rows:
            value = float(data_rows[i][2])
            if value == 0.0:
                value = 0.0  # normalizes -0.0
            data_rows[i][2] = _plain_decimal(value)
        with sub_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "Usage", "predicted_value"])
            writer.writerows(data_rows)
        e_notation_rows = []
        logger.info("Fixup xong — chạy lại check để xác nhận.")
    elif e_notation_rows:
        problems.append(f"{len(e_notation_rows)} ô có e-notation/-0.0 — chạy lại với --fix")

    if problems:
        for p in problems:
            logger.error("CHECK FAIL: %s", p)
        raise SystemExit(1)
    logger.info("CHECK PASS — submission.csv sẵn sàng upload Kaggle.")


# ------------------------------------------------------------------
# judge (optional — requires the app swapped to the judge model)
# ------------------------------------------------------------------

def cmd_judge(args) -> None:
    from src.judge import run_judge

    run_judge(args)


# ------------------------------------------------------------------
# CLI registration
# ------------------------------------------------------------------

def register_cli(sub) -> None:
    p = sub.add_parser("generate", help="Generate candidates (resumable, retrying)")
    p.add_argument("--split", required=True, choices=["validation", "test"])
    p.add_argument("--strategies", default="s4fix")
    p.add_argument("--groups", default="greedy")
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--cache", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int,
                   default=int(os.environ.get("PHASE3_MAX_CONCURRENCY", "24")))
    # (failed records tự động được re-queue ở lần chạy sau — không cần flag riêng)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("repair", help="Guided-JSON repair for broken samples")
    p.add_argument("--split", required=True, choices=["validation", "test"])
    p.add_argument("--cache", required=True)
    p.add_argument("--concurrency", type=int, default=12)
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("score", help="Score an ablation on validation")
    p.add_argument("--split", default="validation", choices=["validation"])
    p.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))
    p.add_argument("--cache", default=str(PHASE3_DIR / "dev_cache.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("judge", help="(OPTIONAL) LLM-judge low-consensus questions")
    p.add_argument("--split", required=True, choices=["validation", "test"])
    p.add_argument("--cache", required=True)
    p.add_argument("--ablation", default="A4", choices=sorted(ABLATIONS))
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("submit", help="Aggregate final values -> test_predictions.json")
    p.add_argument("--cache", default=str(PHASE3_DIR / "test_cache.jsonl"))
    p.add_argument("--ablation", default="A4", choices=sorted(ABLATIONS))
    p.add_argument("--out", default=str(PHASE3_DIR / "test_predictions.json"))
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("check", help="Sanity-gate submission.csv before upload")
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--fix", action="store_true", help="Fix number rendering in place")
    p.set_defaults(func=cmd_check)
