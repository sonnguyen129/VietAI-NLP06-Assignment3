# Data Description — FinQA-VN Self-Evolve Pipeline

This document describes every data file involved in the project: the raw benchmark
(`train.json` / `dev.json` / `test.json`), the program DSL the model must output,
and the auxiliary reasoning dataset (`finqa_qualified_reasonings.json`) used for
few-shot examples and convention reference.

---

## 1. Benchmark files: `train.json`, `dev.json`, `test.json`

**Format:** JSON array of objects. Sizes: train=2,993, dev=584, test=497 (total 4,074).

### 1.1 Top-level item schema

```json
{
  "id": "PM/2017/page_25.pdf-1",
  "pre_text": ["...", "..."],
  "post_text": ["...", "..."],
  "table": [
    ["ngày", "pmi", "nhóm công ty cùng ngành của pmi (1)", "chỉ số s&p 500"],
    ["ngày 31 tháng 12 năm 2012", "$ 100.00", "$ 100.00", "$ 100.00"],
    ["...", "...", "...", "..."]
  ],
  "qa": {
    "question": "Tỷ lệ tăng trưởng giá cổ phiếu của PMI từ năm 2012 đến 2013 là bao nhiêu?",
    "program": "subtract(108.50, 100), divide(#0, 100)",
    "exe_ans": "0.085"
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | `<source>/<year>/<page>.pdf-<n>`. Prefix indicates source: `masvn/...` = Vietnamese stock research (50.8% of data), other tickers (`PM`, `INTC`, `ETR`, etc.) = US filings translated to Vietnamese, `PMI/...` (rare) = additional reports. |
| `pre_text` | list[str] | Paragraphs of narrative text **before** the table. Vietnamese. |
| `post_text` | list[str] | Paragraphs of narrative text **after** the table. Vietnamese. |
| `table` | list[list[str]] | 2D array. **Row 0 is the header row.** Remaining rows are data rows. Avg 8.2 rows, max 29, min 2. All values are strings (may include `$`, `%`, `,` — strip before converting to float). |
| `qa.question` | string | Vietnamese question. |
| `qa.program` | string | **Gold DSL program** — see Section 2. |
| `qa.exe_ans` | string | Gold numeric answer, **as a decimal ratio**, NOT percentage-scaled (e.g. `"0.085"` = 8.5%). One known non-numeric value `"no"` exists in the dataset (treat as an edge case / possible annotation noise). |

### 1.2 Important distributional facts (for eval design / stratification)

- **Source split:** ~50.8% MASVN (Vietnamese reports), ~48.7% US filings (120+ tickers), ~0.5% PMI reports.
- **Operation frequency** (per program, can co-occur): `divide` 58.7%, `subtract` 47.8%, `add` 23.6%, `multiply` 7.9%, `table_max` 5.9%, `table_average` 4.2%, `table_min` 3.3%.
- **Program complexity** (# of chained ops): 1 op = 58.5%, 2 ops = 34.2%, 3 ops = 4.6%, 4 ops = 1.3%, 5 ops = 1.3%, 6–7 ops = 0.2%.
- **Answer value distribution:** ratio/percent (0–1) = 37.2%, large number = 22.1%, small positive = 19.0%, negative = 11.1%, small int = 10.6%.
- **Table-lookup questions** (use `table_*` ops): 14.3%. **Arithmetic-only:** 85.7%.
- **Split complexity drift:** % of multi-step (2+ op) questions: train 43.1% → dev 38.4% → test 35.8%. Test set skews simpler than train.

---

## 2. Program DSL — the exact syntax the model must output

A program is a **flat, comma-separated sequence of function calls**, executed in order.
Results are referenced via `#0`, `#1`, ... (zero-indexed, in the order steps appear).
**No nesting. No quotes. No Python.**

### 2.1 Operation set (8 total)

**Arithmetic (exactly 2 args — number literal or `#N` reference):**

```
add(a, b)        → a + b
subtract(a, b)   → a - b
multiply(a, b)   → a * b
divide(a, b)     → a / b
```

Negative literals are written directly: `add(-167.4, -53.3)`.

**Table lookup (exactly 2 args: `(row_name, col_name_or_none)`):**

```
table_max(row_name, none)
table_min(row_name, none)
table_average(row_name, none)
table_sum(row_name, none)
```

- `row_name` must match a cell in **column 0** of `table` (the row label), **verbatim, no quotes**.
- Second arg is `none` in ~100% of gold programs — operates across all numeric values in that row.

### 2.2 Reference rules

- `#0` = output of step 1, `#1` = output of step 2, etc.
- Later steps can reference any earlier step, not just the immediately preceding one.

### 2.3 Valid examples (from gold data)

```text
# Percentage / growth-rate change (most common pattern)
subtract(108.50, 100), divide(#0, 100)

# Table range
table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1)

# Multi-period average (5 years)
add(v1, v2), add(#0, v3), add(#1, v4), add(#2, v5), divide(#3, 5)

# Two intermediate values both reused later
divide(115.18, 100), divide(113.68, 100), subtract(#0, 1), subtract(#1, 1), add(#2, #3), divide(#4, 2)
```

### 2.4 INVALID patterns the model commonly produces (must be rejected/avoided)

```text
divide(table_average(2013, ...), table_average(2013, ...))   # nesting — not supported
table_average(2013, "tên cột")                                 # quoted args — not supported
table_sum(Danh mục="X", Cột="Y", Bảng=1)                        # keyword args — not supported
table_max(A, B, C, D, E)                                        # >2 args — not supported
```

### 2.5 Parsing implementation notes

- Split top-level program string on `,` **but only at depth-0 parens** (a naive `split(",")` breaks on multi-arg calls — use a paren-aware splitter, or require single-statement-per-comma as gold data does).
- For each step, identify the function name (`re.match(r'^([a-z_]+)\(', step)`), then `rsplit(",", 1)` the inner content into exactly 2 args for arithmetic/table ops.
- Resolve `#N` references against the list of step results computed so far, in order.
- Table row lookup: match `row_name` against `table[i][0]` for `i >= 1` (skip header row); collect all numeric values in `table[i][1:]`, strip `$`, `%`, `,`, whitespace before `float()`.

---

## 3. Answer convention — CRITICAL, must be decided and fixed before prompting

`qa.exe_ans` in train/dev/test is a **raw decimal ratio** (`"0.085"` = 8.5%, NOT `"8.5"`).
Gold programs almost never include `multiply(#X, 100)` (only ~2% of programs).

**This means: the model should output the result of executing the program as-is —
do NOT multiply by 100, even when the question phrasing says "phần trăm" (percent).**
The percent framing is in the *question*, not the *answer scale*.

> Note: the auxiliary `finqa_qualified_reasonings.json` dataset (Section 4) uses the
> **opposite convention** — its `qa.answer` field is percentage-scaled (`"53"` not
> `"0.53"`), and ~52% of its generated programs include `multiply(#X, 100)`. **Do not
> mix conventions.** If using examples from that file for few-shot prompting, either
> (a) strip the `multiply(#X, 100)` step and convert the displayed answer back to
> decimal form to match train/dev/test's convention, or (b) keep a clear internal flag
> so the evaluator knows which convention a given example uses.

Evaluator comparison: compute `exe_val = execute(predicted_program)`, compare to
`float(qa['exe_ans'])` with a relative tolerance (e.g. `abs(exe_val - gold) < 1e-3` or
`abs((exe_val-gold)/gold) < 0.02` for non-zero gold) — gold values have varying decimal
precision (e.g. `"0.085"`, `"1.699"`, `"-3.2"`).

---

## 4. Auxiliary file: `finqa_qualified_reasonings.json`

**Format:** JSON array, 3,247 items. **110 MB — load lazily / stream if memory-constrained.**
This is a *separate, pre-generated reasoning dataset* (not part of train/dev/test) — useful
as a source of vetted few-shot examples and as a smoke-test set with known-good outputs.

### 4.1 Item schema

```json
{
  "filename": "INTC/2013/page_71.pdf",
  "pre_text": [...],
  "post_text": [...],
  "table": "Với ((tính bằng triệu)) '28-12-2013', 'đầu tư nắm giữ để bán' có giá trị là '$ 18086'.\n...",
  "context without markdown": "...",
  "context": "...",
  "context_list": ["### văn bản 1", "...", "### văn bản 2", "..."],
  "qa": {
    "question": "Tỷ lệ phần trăm của tổng tiền và các khoản đầu tư ... là bao nhiêu?",
    "answer": "53",
    "program": "divide(14001, 26302)"
  },
  "verification": {
    "answerable": true,
    "reason": "Ngữ cảnh cung cấp giá trị 14001 ... và 26302 ..., đủ để thực hiện phép chia."
  },
  "verification_time": 3.06,
  "translation_time": 24.27,
  "llm_answers": {
    "reasoning": "Bước 1: ... Bước 2: ... Bước 3: ...",
    "program": "divide(14001, 26302), multiply(#0, 100)",
    "numerical_result": "53.23",
    "raw_response": "Okay, let me try to figure out... <think>...</think>\n\n{\n  \"Reasoning\": \"...\",\n  \"Program syntax\": \"...\",\n  \"Numerical result\": \"...\"\n}"
  }
}
```

| Field | Notes |
|---|---|
| `table` | **Different format from train/dev/test** — here it's a flattened Vietnamese natural-language string ("Với X, Y có giá trị là Z"), not a 2D array. Don't reuse this table format directly if your prompt template expects the array form. |
| `context_list` | Numbered passage chunks (`"### văn bản N"` = "document N") — looks like retrieval/chunking output, possibly for a RAG variant. |
| `qa.answer` | **Percentage-scaled** (see Section 3 — different convention from train/dev/test's `exe_ans`). |
| `verification.answerable` | All 3,247 items are `true` — this file is pre-filtered to only "answerable" questions. |
| `llm_answers.raw_response` | Full model output including `<think>...</think>` CoT block, followed by a JSON object with exactly 3 keys: `"Reasoning"`, `"Program syntax"`, `"Numerical result"`. 3,246/3,247 contain `</think>`. |
| `llm_answers.program` | The model's generated program — 99.3% free of nesting, 0% contain quotes. ~52% include `multiply(#X, 100)` (percent convention). |
| `llm_answers.numerical_result` | Matches `qa.answer` (loose tolerance) in ~96% of items (2,495/2,602 comparable). |

### 4.2 Token budget reference (for setting `max_tokens` in CoT-enabled prompts)

Measured on `llm_answers.raw_response` (chars/4 ≈ tokens):

| Percentile | Tokens |
|---|---|
| p50 | ~1,217 |
| p75 | ~2,340 |
| p90 | ~3,263 |
| p95 | ~3,687 |
| p99 | ~4,259 |
| max | ~5,118 |

→ For any CoT-style prompt (reasoning before final answer), set `max_tokens >= 3000`,
ideally `4096`, to cover p99. Below ~1,024 will truncate >50% of genuine reasoning chains
mid-thought (this was the root cause of the 0% accuracy in earlier self-evo iterations).

---

## 5. Quick-reference: building the prompt + evaluator

**Fixed scaffold (not subject to self-evo mutation):**
1. DSL rules from Section 2 (operation list, arg counts, no-nesting/no-quotes, `#N` refs).
2. Answer-scale convention from Section 3 (decimal ratio, no `*100`).
3. Output format anchor — recommend adopting the `<think>...</think>` + 3-key JSON
   pattern from Section 4.1 if CoT is enabled, so the evaluator can deterministically
   extract `Program syntax` after `</think>`.

**Mutable by self-evo:**
- Instruction phrasing / ordering.
- Which few-shot examples are included (pull from `finqa_qualified_reasonings.json`,
  rewriting `multiply(#X,100)` steps + converting `qa.answer` to decimal to match
  Section 3's convention).
- CoT on/off — **must be paired with the corresponding `max_tokens` setting** (≥3000 if on,
  can be much lower e.g. ~64-128 if off and outputting program only).

**Evaluator:**
- Parse program per Section 2.5, execute against `table` (2D array form, Section 1.1),
  compare to `qa.exe_ans` with tolerance per Section 3.
- Track `output_tokens / num_examples` as a health metric — if it's within ~10% of
  `max_tokens`, flag as truncation/format-collapse before looking at accuracy_by_type.
