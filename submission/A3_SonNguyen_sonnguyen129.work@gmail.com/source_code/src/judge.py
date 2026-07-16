"""
judge.py — (OPTIONAL P7) LLM-as-judge with DeepSeek-R1-Distill-Qwen-7B served
by the SAME Cerebrium app (model swapped sequentially after generation is done).

Key principle: NEVER block the <think> tag. R1 is a reasoning model — it must
think freely before scoring; forcing immediate JSON kills 30-40% of its logic
accuracy. The verdict comes AFTER </think> in a simple regex-parseable format:

    SELECTION: <1-based cluster index, or -1 if all wrong>
    PROGRAM: <corrected flat DSL program, or NONE>
    FINAL_VALUE: <number>

Trust ladder for the verdict: corrected PROGRAM that executes locally >
SELECTION > FINAL_VALUE > fallback to majority vote.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()
PHASE3_DIR = ROOT / "runs" / "phase3"

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
JUDGE_MAX_CALLS = int(os.environ.get("JUDGE_MAX_CALLS", "300"))

_SELECTION_RE = re.compile(r"SELECTION\s*[:=]\s*(-?\d+)", re.IGNORECASE)
_PROGRAM_RE = re.compile(r"PROGRAM\s*[:=]\s*(.+)", re.IGNORECASE)
_VALUE_RE = re.compile(r"FINAL_VALUE\s*[:=]\s*(-?[\d.,eE+-]+)", re.IGNORECASE)


def _make_judge_model():
    """OpenAI-compatible client against the (swapped) Cerebrium endpoint."""
    from src.model import QwenInference

    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("CEREBRIUM_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("CEREBRIUM_API_KEY")
    model = QwenInference(
        model_name_or_path=JUDGE_MODEL,
        base_url=base_url,
        api_key=api_key,
        max_model_len=16384,
    )
    model.load()
    model.assert_served_model(JUDGE_MODEL)  # swap guard — fail fast if still Qwen
    # load() defaults the served-model id to CEREBRIUM_MODEL (Qwen) — force R1.
    model._served_model = JUDGE_MODEL
    return model


def _render_table(table: list[list[str]], max_rows: int = 25) -> str:
    return "\n".join(" | ".join(str(c) for c in row) for row in table[:max_rows])


def build_judge_prompt(row: dict, clusters: list[dict]) -> str:
    """Single user message (R1 convention: no system prompt). Vietnamese content,
    thinking is NOT blocked — the verdict format comes after </think>."""
    lines = [
        "Bạn là giám khảo kiểm tra bài toán tài chính tiếng Việt. "
        "Hãy suy nghĩ từng bước thật kỹ (tự do suy nghĩ), sau đó đưa ra phán quyết cuối cùng.",
        "",
        f"[CÂU HỎI] {row['question']}",
        "",
        f"[NGỮ CẢNH]\n{(row.get('context') or '')[:6000]}",
    ]
    table = row.get("table") or []
    if table:
        lines += ["", f"[BẢNG]\n{_render_table(table)}"]
    lines += [
        "",
        "[QUY TẮC DSL] Chương trình phẳng gồm các hàm add/subtract/multiply/divide/"
        "table_max/table_min/table_sum/table_average(tên_hàng, none), phân cách bằng dấu phẩy, "
        "tham chiếu bước trước bằng #0, #1... KHÔNG lồng hàm. "
        "Đáp án phần trăm ở dạng tỷ lệ thập phân (0.05, KHÔNG phải 5). "
        "'Thay đổi từ A đến B' = giá_trị_B - giá_trị_A (có thể âm).",
        "",
        "[CÁC ỨNG VIÊN]",
    ]
    for i, c in enumerate(clusters, start=1):
        lines.append(
            f"{i}. value={c['value']}  votes={c['weight']}  program: {c.get('program') or 'N/A'}"
        )
    lines += [
        "",
        "Nhiệm vụ: (a) xác định đúng các con số mà câu hỏi yêu cầu; (b) tự tính lại từng ứng viên; "
        "(c) kiểm tra dấu/chiều thay đổi và scale tỷ lệ thập phân; (d) chọn ứng viên đúng, "
        "hoặc nếu tất cả sai thì tự viết chương trình phẳng đúng.",
        "",
        "Sau khi suy nghĩ xong, trả lời CHÍNH XÁC theo định dạng 3 dòng sau (không thêm gì khác):",
        "SELECTION: <số thứ tự ứng viên đúng, hoặc -1 nếu tất cả sai>",
        "PROGRAM: <chương trình phẳng đã sửa, hoặc NONE>",
        "FINAL_VALUE: <giá trị số cuối cùng>",
    ]
    return "\n".join(lines)


def parse_verdict(raw_output: str) -> dict | None:
    """Regex the verdict from the text AFTER the last </think> (or whole text)."""
    text = raw_output.split("</think>")[-1] if "</think>" in raw_output else raw_output
    sel_m = _SELECTION_RE.search(text)
    prog_m = _PROGRAM_RE.search(text)
    val_m = _VALUE_RE.search(text)
    if not (sel_m or prog_m or val_m):
        return None
    program = prog_m.group(1).strip() if prog_m else None
    if program and program.upper().startswith("NONE"):
        program = None
    value = None
    if val_m:
        try:
            value = float(val_m.group(1).replace(",", ""))
        except ValueError:
            value = None
    return {
        "selection": int(sel_m.group(1)) if sel_m else None,
        "corrected_program": program,
        "final_value": value,
    }


def resolve_verdict(verdict: dict, clusters: list[dict], table: list[list[str]]) -> float | None:
    """Trust ladder: executed corrected program > selection > final_value."""
    from src.evaluator import evaluate_program_v2
    from src.extraction_v2 import _validate_program

    program = verdict.get("corrected_program")
    if program:
        validated = _validate_program(program)
        if validated:
            try:
                return float(evaluate_program_v2(validated, table))
            except Exception:
                pass
    selection = verdict.get("selection")
    if selection is not None and 1 <= selection <= len(clusters):
        return float(clusters[selection - 1]["value"])
    value = verdict.get("final_value")
    if value is not None and abs(value) < 1e15:
        return float(value)
    return None


def run_judge(args) -> None:
    from predict import load_split
    from src.pipeline_v2 import (ABLATIONS, PredictionCache, _update_token_usage,
                                 resolve_question)

    config = ABLATIONS[args.ablation]
    rows = {row["id"]: row for row in load_split(args.split)}
    cache = PredictionCache(Path(args.cache))

    # Collect low-consensus questions under the given config (no judge verdicts yet).
    targets = []
    for qid, row in rows.items():
        _, meta = resolve_question(cache, qid, {**config, "judge": False}, {})
        if meta["needs_judge"]:
            targets.append((row, meta["clusters"]))
    logger.info("judge: %d low-consensus questions (cap %d)", len(targets), JUDGE_MAX_CALLS)
    targets = targets[:JUDGE_MAX_CALLS]
    if not targets:
        return

    # Skip already-judged (cache keyed on qid+split).
    judged_path = PHASE3_DIR / "judge_cache.jsonl"
    already = set()
    if judged_path.exists():
        with judged_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    already.add((d.get("qid"), d.get("split")))
                except Exception:
                    continue
    targets = [(row, clusters) for row, clusters in targets
               if (row["id"], args.split) not in already]
    logger.info("judge: %d remaining after cache", len(targets))
    if not targets:
        return

    model = _make_judge_model()

    def judge_one(item):
        row, clusters = item
        prompt_text = build_judge_prompt(row, clusters)
        # R1 chat template: single user turn, thinking NOT blocked.
        prompt = model.format_prompt(system_message="", user_message=prompt_text,
                                     enable_thinking=True)
        try:
            gens = model.complete_many(
                prompt, n=1, temperature=0.6, top_p=0.95, top_k=-1,
                repetition_penalty=1.0, max_new_tokens=4096, max_attempts=4,
            )
            return row, clusters, gens[0], None
        except Exception as exc:
            return row, clusters, None, exc

    from tqdm import tqdm

    total_in = total_out = ok = failed = 0
    with judged_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(judge_one, t) for t in targets]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="judge"):
                row, clusters, gen, exc = fut.result()
                if exc is not None or gen is None:
                    logger.error("judge failed qid=%s: %s", row["id"], exc)
                    failed += 1
                    continue
                total_in += gen.input_tokens
                total_out += gen.output_tokens
                verdict = parse_verdict(gen.raw_output)
                final_value = None
                if verdict:
                    final_value = resolve_verdict(verdict, clusters, row.get("table") or [])
                record = {
                    "qid": row["id"], "split": args.split,
                    "final_value": final_value,
                    "verdict": verdict,
                    "clusters": clusters,
                    "think_excerpt": gen.raw_output[:2000],
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                ok += int(final_value is not None)
                failed += int(final_value is None)

    _update_token_usage(total_in, total_out, "judge_r1")
    logger.info("judge done: resolved=%d unresolved=%d tokens in=%d out=%d",
                ok, failed, total_in, total_out)
