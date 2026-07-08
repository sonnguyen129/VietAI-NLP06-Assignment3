# FinQA DSL Syntax Rules

## Core Structure
A program is a **comma-separated, flat list of function calls** executed top to bottom. No nesting. No Python. No imports.

```
step1, step2, step3, ...
```

Results from previous steps are referenced as `#0`, `#1`, `#2`, etc. (zero-indexed).

## Arithmetic Operations
These take **exactly 2 arguments** — either numeric literals or `#N` references:

```
add(a, b)        → a + b
subtract(a, b)   → a - b
multiply(a, b)   → a * b
divide(a, b)     → a / b
```

**Numeric literals:** plain numbers, no quotes. Negatives are fine.
```
subtract(108.50, 100)           ✅
add(-167.4, -53.3)              ✅
divide(#0, 100)                 ✅ (reference to step 0 result)
divide(#0, #1)                  ✅ (reference both)
subtract("108.50", "100")       ❌ no quotes
```

## Table Operations
These take **exactly 2 arguments**: `(row_name, col_name)`. When you want the full row (all columns), use `none` as the column.

```
table_max(row_name, col_name_or_none)
table_min(row_name, col_name_or_none)
table_average(row_name, col_name_or_none)
table_sum(row_name, col_name_or_none)
```

**The row_name must match the table header exactly — no quotes, no extra spaces:**
```
table_max(Lãi ròng, none)                   ✅
table_average(EPS (VND), none)              ✅
table_max(LNHĐKD, none), table_min(LNHĐKD, none), subtract(#0, #1)   ✅ chained
table_average("doanh thu thuần", none)      ❌ no quotes
table_max(Lãi ròng, EPS, ROE)              ❌ only 2 args allowed
```

In this dataset, the column arg is **always `none`** — meaning operate across all values in that row.

## Reference Rules
- `#0` = result of step 1, `#1` = result of step 2, etc.
- You can reference any prior step, not just the immediately previous one.

## Full Operation List
| Operation | Args |
|---|---|
| `add` | `(num_or_ref, num_or_ref)` |
| `subtract` | `(num_or_ref, num_or_ref)` |
| `multiply` | `(num_or_ref, num_or_ref)` |
| `divide` | `(num_or_ref, num_or_ref)` |
| `table_max` | `(row_name, none)` |
| `table_min` | `(row_name, none)` |
| `table_average` | `(row_name, none)` |
| `table_sum` | `(row_name, none)` |

## What to Tell the Model
Include the following in the system prompts to constrain the LLM into generating valid DSL:

```
Quy tắc viết chương trình:
1. Mỗi bước là một hàm riêng biệt, phân cách bằng dấu phẩy
2. KHÔNG lồng hàm vào nhau (không dùng divide(table_average(...), ...))
3. Dùng #0, #1, #2... để tham chiếu kết quả của bước trước (bắt đầu từ #0)
4. Tên cột/hàng trong table_xxx KHÔNG dùng dấu ngoặc kép
5. table_xxx chỉ nhận đúng 2 tham số: (tên_hàng, none)
6. Số âm viết trực tiếp: add(-167.4, -53.3) — không dùng ngoặc thêm
7. CỰC KỲ QUAN TRỌNG: Nếu cần một giá trị cụ thể từ bảng (ví dụ: doanh thu năm 2022), KHÔNG dùng hàm table_xxx. Hãy tự đọc bảng và viết TRỰC TIẾP con số đó vào hàm toán học.

Ví dụ đúng:
  subtract(108.50, 100), divide(#0, 100) (Trích xuất trực tiếp số 108.50 và 100 từ văn bản/bảng)
  table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1) (Tính max/min trên toàn bộ hàng)
  divide(115.18, 100), divide(113.68, 100), subtract(#0, 1), subtract(#1, 1), add(#2, #3), divide(#4, 2)
```
