"""
extraction_v2.py — Phase-3 robust program extraction + repair prompts.

Replaces the greedy legacy `extract_answer` behaviour (which concatenated every
op(...) fragment found anywhere in the monologue) with a strict priority
ladder. Never scavenges fragments out of a runaway thinking loop — those go to
the guided-JSON repair pass instead.

The legacy `extract_answer` in src/model.py is left untouched (grader surface).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# The 8 official DSL ops plus the extras the evaluator accepts.
OPS = {
    "add", "subtract", "multiply", "divide", "exp", "greater", "abs",
    "table_max", "table_min", "table_sum", "table_average",
}

_PROGRAM_JSON_KEYS = (
    "Program syntax", "program syntax", "Program", "program", "program_syntax",
)
_RESULT_JSON_KEYS = ("Numerical result", "numerical result", "result", "Result")

MAX_STEPS = 8  # anything longer is runaway concatenation, not a real program


@dataclass
class ExtractionResult:
    program: Optional[str]
    stated_result: Optional[float]
    tag: str  # "json_after_think" | "json_anywhere" | "line_scan" | "runaway" | "empty" | "none"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_json_blocks(text: str) -> list[str]:
    """Return balanced {...} blocks found in text, in order of appearance."""
    blocks = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"' and depth > 0:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    blocks.append(text[start:i + 1])
                    start = -1
    return blocks


def _parse_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        s = str(value).strip().replace("%", "").replace(",", "")
        if not s:
            return None
        try:
            result = float(s)
        except ValueError:
            return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _program_from_json_text(json_text: str) -> tuple[Optional[str], Optional[float]]:
    try:
        data = json.loads(json_text, strict=False)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None

    program = None
    for key in _PROGRAM_JSON_KEYS:
        if key in data and isinstance(data[key], str):
            program = data[key]
            break
    if program is None:
        for key, val in data.items():
            if isinstance(val, str) and ("program" in key.lower() or "syntax" in key.lower()):
                program = val
                break

    stated = None
    for key in _RESULT_JSON_KEYS:
        if key in data:
            stated = _parse_number(data[key])
            break
    if stated is None:
        for key, val in data.items():
            if "result" in key.lower() or "kết quả" in key.lower():
                stated = _parse_number(val)
                break

    if program is not None:
        program = re.sub(r"^`+", "", program.strip())
        program = re.sub(r"`+$", "", program).strip()
    return (program or None), stated


def _split_top_level_steps(program: str) -> list[str]:
    """Split a flat program on top-level commas that separate op calls.

    'subtract(1, 2), divide(#0, 2)' -> ['subtract(1, 2)', 'divide(#0, 2)']
    Commas inside parentheses are preserved.
    """
    steps = []
    depth = 0
    current = []
    for ch in program:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            step = "".join(current).strip()
            if step:
                steps.append(step)
            current = []
        else:
            current.append(ch)
    step = "".join(current).strip()
    if step:
        steps.append(step)
    return steps


_STEP_RE = re.compile(r"^([a-zA-Z_][a-zA-Z_0-9]*)\s*\((.*)\)$", re.DOTALL)


def _validate_program(program: str) -> Optional[str]:
    """Normalize and validate a candidate program. Return cleaned program or None."""
    from src.model import clean_and_fix_program

    if not program:
        return None
    program = clean_and_fix_program(program.strip().strip(","))
    if not program:
        return None

    steps = _split_top_level_steps(program)
    if not steps or len(steps) > MAX_STEPS:
        return None

    # Dedupe consecutive identical steps (runaway repetition artefact).
    deduped: list[str] = []
    for step in steps:
        if not deduped or deduped[-1] != step:
            deduped.append(step)
    steps = deduped

    for idx, step in enumerate(steps):
        m = _STEP_RE.match(step)
        if not m:
            return None
        op = m.group(1).lower()
        if op not in OPS:
            return None
        args_str = m.group(2)
        if "\n" in args_str or len(args_str) > 120:
            return None
        if op.startswith("table_"):
            continue  # row-name args are free text
        # Non-table ops: every arg must be a number or a valid #ref.
        for arg in args_str.split(","):
            arg = arg.strip()
            if not arg:
                return None
            if arg.startswith("#"):
                try:
                    ref = int(arg[1:])
                except ValueError:
                    return None
                if ref >= idx:
                    return None
            else:
                cleaned = arg.replace("%", "").replace(",", "")
                try:
                    float(cleaned)
                except ValueError:
                    return None

    return ", ".join(steps)


# Truncated-JSON fallback: the output hit the token cap before the closing
# brace, so no balanced block exists — pull the key values straight out.
_KEY_PROGRAM_RE = re.compile(r'"[Pp]rogram[ _]?syntax"\s*:\s*"([^"\n]+)"')
_KEY_RESULT_RE = re.compile(r'"[Nn]umerical[ _]?result"\s*:\s*"?\s*(-?[\d.,eE+]+)')


def _program_from_truncated_json(text: str) -> tuple[Optional[str], Optional[float]]:
    matches = _KEY_PROGRAM_RE.findall(text)
    if not matches:
        return None, None
    program = matches[-1].strip()
    stated = None
    result_matches = _KEY_RESULT_RE.findall(text)
    if result_matches:
        stated = _parse_number(result_matches[-1])
    return program, stated


_PROGRAM_CHARS_RE = re.compile(r"[a-zA-Z_]+")


def _line_is_program_like(line: str) -> bool:
    """A line qualifies only if every alphabetic token is a known op or 'none'."""
    if "(" not in line or ")" not in line:
        return False
    for token in _PROGRAM_CHARS_RE.findall(line):
        if token.lower() not in OPS and token.lower() != "none":
            return False
    return True


# ------------------------------------------------------------------
# Main extraction ladder
# ------------------------------------------------------------------

def extract_program_v2(raw_output: str) -> ExtractionResult:
    if not raw_output or not raw_output.strip():
        return ExtractionResult(None, None, "empty")

    text = raw_output.strip()

    # Runaway: the thinking block never closed. Do NOT scavenge fragments —
    # that is exactly what produced the crash-concatenations. Repair instead.
    if "</think>" not in text:
        return ExtractionResult(None, None, "runaway")

    after_think = text.split("</think>")[-1]

    # 1. JSON after the last </think> (the healthy path — covers ~100% of
    #    well-terminated outputs). Last block wins.
    for block in reversed(_find_json_blocks(after_think)):
        program, stated = _program_from_json_text(block)
        validated = _validate_program(program) if program else None
        if validated:
            return ExtractionResult(validated, stated, "json_after_think")

    # 2. Truncated JSON after </think> (hit the token cap before the closing
    #    brace): pull "Program syntax" straight out with a regex.
    program, stated = _program_from_truncated_json(after_think)
    if program:
        validated = _validate_program(program)
        if validated:
            return ExtractionResult(validated, stated, "json_truncated")

    # 3. Any JSON anywhere in the output (model wrapped it oddly / emitted early).
    for block in reversed(_find_json_blocks(text)):
        program, stated = _program_from_json_text(block)
        validated = _validate_program(program) if program else None
        if validated:
            return ExtractionResult(validated, stated, "json_anywhere")

    # 3. Bottom-up line scan: take the LAST program-pure line only.
    for line in reversed(after_think.splitlines()):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:PROGRAM|Chương trình|Program)\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^`+|`+$", "", line).strip()
        if not _line_is_program_like(line):
            continue
        validated = _validate_program(line)
        if validated:
            return ExtractionResult(validated, None, "line_scan")

    return ExtractionResult(None, None, "none")


# ------------------------------------------------------------------
# Repair pass prompt
# ------------------------------------------------------------------

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"Program syntax": {"type": "string"}},
    "required": ["Program syntax"],
}

_REPAIR_SYSTEM = (
    "Bạn là trợ lý AI chuyên phân tích tài chính tiếng Việt. Nhiệm vụ: viết lại "
    "MỘT chương trình DSL phẳng, đúng cú pháp, trả lời câu hỏi dựa trên ngữ cảnh."
)


def build_repair_message(question: str, context: str, raw_tail: str) -> str:
    """User message for the guided-JSON repair pass."""
    from src.executor import _DSL_BLOCK

    parts = [
        f"Bối cảnh:\n{context}",
        f"Câu hỏi: {question}",
    ]
    if raw_tail:
        parts.append(
            "Bản nháp trước (có thể sai cú pháp hoặc bị cắt giữa chừng — chỉ dùng tham khảo):\n"
            f"{raw_tail[-1200:]}"
        )
    parts.append(_DSL_BLOCK)
    parts.append(
        "Hãy trả về DUY NHẤT một khối JSON dạng {\"Program syntax\": \"<chương trình phẳng>\"}. "
        "Chương trình phải phẳng (KHÔNG lồng hàm), các bước phân cách bằng dấu phẩy, "
        "tham chiếu bước trước bằng #0, #1..."
    )
    return "\n\n".join(parts)


def repair_system_message() -> str:
    return _REPAIR_SYSTEM
