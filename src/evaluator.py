import re
from typing import List

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from a table cell string."""
    text = text.replace(',', '')
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    return [float(m) for m in matches]

def evaluate_program(program: str, table: List[List[str]]) -> float:
    """
    Evaluate a FinQA-style program.
    E.g. "subtract(108.50, 100), divide(#0, 100)"
    or "table_max(chênh lệch tỷ giá hối đoái, none)"
    """
    if not program:
        raise ValueError("Empty program")
        
    # FinQA format uses commas to separate steps, but arguments can be strings containing commas or parentheses.
    steps = []
    current_op = ""
    current_args = ""
    in_args = False
    paren_depth = 0
    i = 0
    while i < len(program):
        c = program[i]
        if not in_args:
            if c == '(':
                in_args = True
                paren_depth = 1
            elif c.isalnum() or c == '_':
                current_op += c
        else:
            if c == '(':
                paren_depth += 1
                current_args += c
            elif c == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    steps.append((current_op.strip(), current_args.strip()))
                    current_op = ""
                    current_args = ""
                    in_args = False
                    while i + 1 < len(program) and program[i+1] in [',', ' ']:
                        i += 1
                else:
                    current_args += c
            else:
                current_args += c
        i += 1
    results = []
    
    for op, args_str in steps:
        op = op.lower().strip()
        
        if op.startswith("table_"):
            # Table ops take 2 arguments, often strings with commas.
            # Usually format is "row_name, none" or "none, col_name"
            if args_str.lower().endswith(", none"):
                arg1 = args_str[:-6].strip()
                arg2 = "none"
            elif args_str.lower().startswith("none,"):
                arg1 = "none"
                arg2 = args_str[5:].strip()
            else:
                parts = args_str.rsplit(",", 1)
                arg1 = parts[0].strip() if len(parts) > 0 else ""
                arg2 = parts[1].strip() if len(parts) > 1 else ""
            args = [arg1, arg2]
        else:
            args = [arg.strip() for arg in args_str.split(",")]
        
        resolved_args = []
        for arg in args:
            if arg.startswith("#"):
                try:
                    idx = int(arg[1:])
                    resolved_args.append(results[idx])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid reference {arg}")
            elif arg.lower() == "none":
                resolved_args.append(None)
            else:
                # Try float
                clean_arg = arg.replace("%", "").replace(",", "")
                try:
                    resolved_args.append(float(clean_arg))
                except ValueError:
                    resolved_args.append(arg)
                    
        # Evaluate operation
        try:
            if op == "add":
                res = resolved_args[0] + resolved_args[1]
            elif op == "subtract":
                res = resolved_args[0] - resolved_args[1]
            elif op == "multiply":
                res = resolved_args[0] * resolved_args[1]
            elif op == "divide":
                res = resolved_args[0] / resolved_args[1]
            elif op == "exp":
                res = resolved_args[0] ** resolved_args[1]
            elif op == "greater":
                res = resolved_args[0] if resolved_args[0] > resolved_args[1] else resolved_args[1]
            elif op == "abs":
                res = abs(resolved_args[0])
            elif op.startswith("table_"):
                row_name = resolved_args[0] if isinstance(resolved_args[0], str) else None
                col_name = resolved_args[1] if len(resolved_args) > 1 and isinstance(resolved_args[1], str) else None
                
                numbers = []
                if row_name and row_name.lower() != "none" and table:
                    row_idx = -1
                    for i, row in enumerate(table):
                        if row and row_name.lower() in row[0].lower():
                            row_idx = i
                            break
                    if row_idx != -1:
                        for cell in table[row_idx][1:]:
                            numbers.extend(extract_numbers(cell))
                elif col_name and col_name.lower() != "none" and table:
                    if table and table[0]:
                        col_idx = -1
                        for j, header in enumerate(table[0]):
                            if col_name.lower() in header.lower():
                                col_idx = j
                                break
                        if col_idx != -1:
                            for i in range(1, len(table)):
                                if col_idx < len(table[i]):
                                    numbers.extend(extract_numbers(table[i][col_idx]))
                                    
                if not numbers:
                    raise ValueError(f"No numbers found for table operation {op}")
                    
                if op == "table_max":
                    res = max(numbers)
                elif op == "table_min":
                    res = min(numbers)
                elif op == "table_sum":
                    res = sum(numbers)
                elif op == "table_average":
                    res = sum(numbers) / len(numbers)
                else:
                    raise ValueError(f"Unknown table op {op}")
            else:
                raise ValueError(f"Unknown op {op}")
        except Exception as e:
            raise ValueError(f"Error evaluating {op} with args {resolved_args}: {e}")
            
        results.append(res)
        
    if not results:
        raise ValueError("No valid steps parsed from program")
        
    return float(results[-1])

# ======================================================================
# Phase-3 hardened variants (add-only; the originals above stay untouched)
# ======================================================================

import math
import unicodedata


def extract_numbers_v2(text: str) -> List[float]:
    """
    Accounting-aware number extraction.

    Financial tables often render negatives as "$ -61.1 ( 61.1 )" — the
    parenthesized value is a duplicate magnitude, not a second number. The
    legacy extract_numbers returns [-61.1, 61.1], which makes table_max pick
    the wrong sign. Rules:
      - "( x )" right after a number of the same magnitude -> drop the duplicate.
      - A cell that is ONLY "( x )" is accounting notation for a negative -> [-x].
    """
    raw = text.replace(",", "")

    # Cell that is purely parenthesized: accounting negative.
    only_paren = re.fullmatch(r"\s*\(\s*([-+]?\d*\.?\d+)\s*\)\s*%?\s*", raw)
    if only_paren:
        return [-abs(float(only_paren.group(1)))]

    numbers: List[float] = []
    skip_next_value: float | None = None
    # Tokenize into (number, in_parens) pairs preserving order.
    for match in re.finditer(r"(\(\s*[-+]?\d*\.?\d+\s*\))|([-+]?\d*\.?\d+)", raw):
        if match.group(1) is not None:
            inner = float(re.sub(r"[()\s]", "", match.group(1)))
            if skip_next_value is not None and abs(abs(skip_next_value) - abs(inner)) <= 1e-9:
                skip_next_value = None
                continue  # duplicate magnitude of the preceding negative
            numbers.append(inner)
            skip_next_value = None
        else:
            value = float(match.group(2))
            numbers.append(value)
            skip_next_value = value if value < 0 else None
    return numbers


def _normalize_row_name(name: str) -> str:
    name = name.casefold().strip()
    # Strip unit suffixes like "(x)", "(%)", "(lần)", "(usd)", "(tỷ đồng)".
    name = re.sub(r"\s*\([^)]{0,20}\)\s*$", "", name)
    # Strip diacritics.
    name = unicodedata.normalize("NFD", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Mn")
    # Drop punctuation.
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def fuzzy_row_match(row_name: str, table: List[List[str]]) -> int | None:
    """Tiered row lookup: exact -> substring (both directions) -> normalized -> Jaccard."""
    if not table or not row_name:
        return None
    target = row_name.lower().strip()

    # Tier 1: exact (case-insensitive).
    for i, row in enumerate(table):
        if row and row[0].lower().strip() == target:
            return i
    # Tier 2: legacy substring, both directions.
    for i, row in enumerate(table):
        if row and (target in row[0].lower() or row[0].lower().strip() in target and row[0].strip()):
            return i
    # Tier 3: normalized (diacritics/punctuation/unit-suffix insensitive).
    norm_target = _normalize_row_name(row_name)
    if norm_target:
        for i, row in enumerate(table):
            if row and _normalize_row_name(row[0]) == norm_target:
                return i
        for i, row in enumerate(table):
            row_norm = _normalize_row_name(row[0]) if row else ""
            if row_norm and (norm_target in row_norm or row_norm in norm_target):
                return i
        # Tier 4: token-set Jaccard >= 0.6, best row wins.
        target_tokens = set(norm_target.split())
        best_i, best_score = None, 0.0
        if target_tokens:
            for i, row in enumerate(table):
                row_tokens = set(_normalize_row_name(row[0]).split()) if row else set()
                if not row_tokens:
                    continue
                jaccard = len(target_tokens & row_tokens) / len(target_tokens | row_tokens)
                if jaccard > best_score:
                    best_i, best_score = i, jaccard
            if best_score >= 0.6:
                return best_i
    return None


_CALL_HEAD_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*\(")

_KNOWN_OPS_V2 = {
    "add", "subtract", "multiply", "divide", "exp", "greater", "abs",
    "table_max", "table_min", "table_sum", "table_average",
}


def _split_args_top_level(args_str: str) -> List[str]:
    args, depth, current = [], 0, []
    for ch in args_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def flatten_nested_program(program: str) -> str:
    """
    Rewrite nested calls into flat SSA steps:
        divide(subtract(500, 400), 400) -> subtract(500, 400), divide(#0, 400)
    Already-flat programs (with or without #refs) pass through unchanged.
    """
    if not program:
        return program

    # Split into original top-level steps.
    steps = _split_args_top_level(program)  # top-level comma split works for steps too
    # Re-join fragments that are not calls (e.g. table row names containing commas
    # would already be inside parens, so each fragment here should be a call).
    out_steps: List[str] = []
    orig_to_new: dict = {}

    def emit(op: str, args: List[str]) -> int:
        out_steps.append(f"{op}({', '.join(args)})")
        return len(out_steps) - 1

    def flatten_call(expr: str) -> int | None:
        m = _CALL_HEAD_RE.match(expr)
        if not m or not expr.rstrip().endswith(")"):
            return None
        op = m.group(1).lower()
        if op not in _KNOWN_OPS_V2:
            return None
        inner = expr[m.end():expr.rstrip().rfind(")")]
        raw_args = _split_args_top_level(inner)
        new_args: List[str] = []
        for arg in raw_args:
            sub = _CALL_HEAD_RE.match(arg)
            if sub and sub.group(1).lower() in _KNOWN_OPS_V2 and arg.rstrip().endswith(")"):
                ref_idx = flatten_call(arg)
                if ref_idx is None:
                    return None
                new_args.append(f"#{ref_idx}")
            elif arg.startswith("#"):
                try:
                    orig_ref = int(arg[1:])
                except ValueError:
                    return None
                new_args.append(f"#{orig_to_new.get(orig_ref, orig_ref)}")
            else:
                new_args.append(arg)
        return emit(op, new_args)

    for orig_idx, step in enumerate(steps):
        new_idx = flatten_call(step)
        if new_idx is None:
            return program  # unparseable — leave to the caller's error handling
        orig_to_new[orig_idx] = new_idx

    return ", ".join(out_steps)


def evaluate_program_v2(program: str, table: List[List[str]]) -> float:
    """
    Hardened executor: nested-call flattening + fuzzy row match + accounting-
    aware numbers + div-by-zero / non-finite guards. Same DSL semantics as
    evaluate_program otherwise.
    """
    if not program:
        raise ValueError("Empty program")

    program = flatten_nested_program(program)
    steps = _split_args_top_level(program)
    if not steps:
        raise ValueError("No valid steps parsed from program")

    results: List[float] = []
    for step in steps:
        m = _CALL_HEAD_RE.match(step)
        if not m or not step.rstrip().endswith(")"):
            raise ValueError(f"Malformed step: {step!r}")
        op = m.group(1).lower()
        args_str = step[m.end():step.rstrip().rfind(")")]

        if op.startswith("table_"):
            if args_str.lower().rstrip().endswith("none") and "," in args_str:
                arg1 = args_str[:args_str.rfind(",")].strip()
            else:
                arg1 = args_str.strip()
            row_idx = fuzzy_row_match(arg1, table)
            numbers: List[float] = []
            if row_idx is not None:
                for cell in table[row_idx][1:]:
                    numbers.extend(extract_numbers_v2(cell))
            if not numbers:
                raise ValueError(f"No numbers found for table operation {op}({arg1!r})")
            if op == "table_max":
                res = max(numbers)
            elif op == "table_min":
                res = min(numbers)
            elif op == "table_sum":
                res = sum(numbers)
            elif op == "table_average":
                res = sum(numbers) / len(numbers)
            else:
                raise ValueError(f"Unknown table op {op}")
        else:
            resolved: List[float] = []
            for arg in _split_args_top_level(args_str):
                arg = arg.strip()
                if arg.startswith("#"):
                    try:
                        idx = int(arg[1:])
                        resolved.append(results[idx])
                    except (ValueError, IndexError):
                        raise ValueError(f"Invalid reference {arg}")
                else:
                    clean_arg = arg.replace("%", "").replace(",", "")
                    try:
                        resolved.append(float(clean_arg))
                    except ValueError:
                        raise ValueError(f"Non-numeric argument {arg!r} for op {op}")
            if op == "add":
                res = resolved[0] + resolved[1]
            elif op == "subtract":
                res = resolved[0] - resolved[1]
            elif op == "multiply":
                res = resolved[0] * resolved[1]
            elif op == "divide":
                if resolved[1] == 0:
                    raise ValueError("Division by zero")
                res = resolved[0] / resolved[1]
            elif op == "exp":
                res = resolved[0] ** resolved[1]
            elif op == "greater":
                res = resolved[0] if resolved[0] > resolved[1] else resolved[1]
            elif op == "abs":
                res = abs(resolved[0])
            else:
                raise ValueError(f"Unknown op {op}")

        if not math.isfinite(res):
            raise ValueError(f"Non-finite result from {op}")
        results.append(float(res))

    return float(results[-1])


if __name__ == "__main__":
    # Test cases
    print(evaluate_program("subtract(108.50, 100), divide(#0, 100)", [])) # 0.085
    table = [
        ['', '2015', '2014'],
        ['chênh lệch tỷ giá', '$ -61.1 ( 61.1 )', '$ -16.6 ( 16.6 )']
    ]
    print(evaluate_program("table_max(chênh lệch tỷ giá, none)", table)) # -16.6
