# 📋 Kế hoạch Giai đoạn 2 (Phase 3 Kaggle): 0.599 → mục tiêu ≥ 0.78

> Tài liệu nghiên cứu — tổng hợp từ forensics trên artifacts thật của giai đoạn 1.
> Baseline LB: **0.59919** · Top LB: **0.80971** · Deadline: **2026-07-09 23:59 ICT**
> Ràng buộc: giữ nguyên serving Cerebrium (A10, vLLM 0.24, Qwen3.5-4B-AWQ) · chỉ còn 1 lần submit.

---

## 1. Forensics — điểm đang rơi ở đâu (bằng chứng thật, không đoán)

Phân tích `runs/exp_self/iter_004_eval_dev.json` (dev 240, acc 0.579) và `submission_details.json` (test 494):

| # | Vấn đề | Bằng chứng | Điểm rơi ước tính |
|---|--------|-----------|-------------------|
| 1 | **Runaway CoT** — model không bao giờ đóng `</think>`, suy nghĩ lặp vô hạn đến hết 4096 token | 51/240 dev → **0% đúng** trên nhóm này; 64/494 test | ~50% tổng lỗi dev |
| 2 | **46/494 test (9.3%) raw_output RỖNG** — request fail khi pod restart, `generate_batch` nuốt exception trả `""` → predicted_value 0.0 | 46 record empty trong submission_details.json | ~9% test bị 0 điểm oan |
| 3 | **`extract_answer` gom rác** — nối MỌI mảnh `op(...)` xuất hiện trong cả bài monologue thành 1 "chương trình" | 47/240 dev crash evaluator (prose thành op: 20, nested: 13, table row sai: 8, #ref sai: 3) | gần nửa số lỗi dev |
| 4 | **Few-shot #4 của iter_004 dạy SAI cú pháp** — `divide(subtract(500,400),400)` là nested call mà chính evaluator của đề không chạy được | 28 chương trình nested trên test → tất cả 0.0 | ~5.5% test |
| 5 | **Lỗi dấu / chiều thay đổi** — "tăng/giảm từ A đến B" bị đảo operand | 7 dev case pred = −gold | ~3% dev |
| 6 | **Sai reasoning thật** — thiếu bước chia % cuối, đọc sai số từ bảng, sai mẫu số | ~35 dev case "far off" | ~15% dev |
| 7 | **`table_op` tệ nhất** — substring row match fail ("P/E" vs "P/E (x)"); accounting negative `"$ -61.1 ( 61.1 )"` bị đọc thành [−61.1, **+61.1**] → table_max chọn sai dấu | acc 32.5%, crash 47.5% | loại câu yếu nhất |

**Phát hiện vàng 🔑:** trong 384/494 output test có đóng `</think>` tử tế, **100% đều chứa khối JSON `"Program syntax"` parse được**. Tức là chỉ cần (a) ép model đóng think + (b) extraction ưu tiên JSON-sau-think, là giải quyết gần trọn nhóm lỗi #1 và #3.

**Nguyên nhân gốc của runaway:** proof run chạy `temperature=0.0` (greedy) và code nhánh temp=0 **không có repetition penalty** — đúng điều kiện sinh vòng lặp "Tuy nhiên… Nhưng chờ đã…". Reflection R3 của chính EvoAgent đã chẩn đoán đúng bệnh này nhưng iteration 4 không áp dụng.

**Dữ liệu:**
- `train.json`: 2986 câu **có gold program + exe_ans**, và gold **100% flat** (không nested) → nguồn few-shot chuẩn miễn phí.
- `dev.json`: 584 câu có gold — proof run mới dùng slice 240. **Giai đoạn 2 validate trên full 584.**
- `test.json`: 494 câu, **KHÔNG có gold** → không tự chấm test được, chỉ có dev làm la bàn.

**Luật Kaggle (từ docs/PHASE3_KAGGLE.md):** cho phép fine-tune, model ≤9B, external API (nếu tài liệu hóa trong report), ensembling/self-consistency/retrieval được khuyến khích. Metric chính xác không ghi trong repo → dùng scorer nội bộ `abs(pred−gold) ≤ 1e-4` làm chuẩn và format giá trị submit thật cẩn thận.

---

## 2. Chiến lược tổng thể

Hai mũi tấn công, xếp theo độ chắc chắn:

**Mũi 1 — Vá các lỗ rò (deterministic, không cần thêm GPU nhiều, ~+15-20 điểm):**
- Retry + cache + resume cho inference (yêu cầu #3 của bạn) → cứu 9.3% câu bị 0 oan.
- Fix decoding (temp 0.6 / top_p 0.95 / top_k 20 / repetition_penalty 1.05 theo khuyến nghị Qwen cho thinking mode) → diệt runaway.
- Extraction v2 (JSON-sau-think, không mót rác) + Repair pass (re-ask guided JSON cho câu hỏng).
- Sửa few-shot #4 + làm cứng evaluator (hỗ trợ nested bằng auto-flatten, fuzzy row match, accounting negatives).

**Mũi 2 — Booster self-consistency + ensemble (~+5-10 điểm):**
- **Self-consistency**: K=8 sample/câu (1 greedy + 7 sampled), thực thi từng chương trình, **vote theo GIÁ TRỊ đã execute** (cluster với tolerance).
- **Ensemble 2 strategies (cùng nền s4fix, chỉ khác nguồn few-shot)**: (A) few-shot cố định đã sửa × (B) retrieval-few-shot per-question (kNN từ train). Cùng base template → chỉ biến đổi MỘT biến (nguồn shot), dễ quy kết nguyên nhân.
- **(OPTIONAL — stretch, NGOÀI critical path) LLM-as-judge**: các câu bất đồng thuận (~20-35%) đưa cho **DeepSeek-R1-Distill-Qwen-7B** phán xử. **Submission mặc định KHÔNG phụ thuộc judge** (xem §3.7) — chỉ bật nếu đã có submission an toàn + còn dư thời gian/GPU.

**Loại khỏi phạm vi:** fine-tuning (đòi đổi serving config — vi phạm ràng buộc của bạn — và không kịp deadline).

---

## 3. Kiến trúc & file

```
assignment03/
├── predict.py                    # MỚI — CLI giai đoạn 2: rescore/generate/repair/score/submit/check
├── src/model.py                  # SỬA (chỉ THÊM): complete_many() = n>1 + retry/backoff
├── src/evaluator.py              # SỬA (chỉ THÊM): evaluate_program_v2 (+flatten/fuzzy/accounting)
├── src/extraction_v2.py          # MỚI — extract_program_v2 + prompt repair
├── src/pipeline_v2.py            # MỚI — JSONL cache/resume + as_completed scheduler
├── src/retrieval.py              # MỚI — TF-IDF kNN few-shot (sklearn, không cần torch)
├── src/voting.py                 # MỚI — cluster giá trị + weighted vote
├── src/judge.py                  # MỚI (OPTIONAL) — judge R1-Distill (cùng app Cerebrium, swap model tuần tự), fallback majority
├── runs/phase3/                  # MỚI — toàn bộ artifacts giai đoạn 2 (xem §3.9 Logging & artifacts)
├── run_cerebrium.py              # SỬA (chỉ THÊM): subcommand predict-dev / predict-test
├── format_submission.py          # DÙNG LẠI NGUYÊN TRẠNG — sinh submission.csv đúng schema Kaggle (id,Usage,predicted_value)
└── .env                          # THÊM: JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_MAX_CALLS, PHASE3_MAX_CONCURRENCY
```

**Bất khả xâm phạm** (bảo toàn điểm giai đoạn 1): `graders/*`, `evolution_proof.json`, `runs/exp_self/*`, `src/harness.py`, `src/executor.py`, `src/sandbox.py`, `self_proposer/reflector.py`, `submit.py`, `format_submission.py`. Mọi thay đổi vào `model.py`/`evaluator.py` là **add-only** — hàm cũ giữ nguyên từng byte. Sau khi xong phải re-run đủ 6 graders.

Cài thêm local: `pip install scikit-learn` (son_env — judge gọi qua endpoint OpenAI-compatible của vLLM nên tái dùng SDK `openai` đã có sẵn, không cần cài thêm gì).

### 3.1 Reliability layer (yêu cầu #3)

- `QwenInference.complete_many(prompt, n, temperature, ..., max_attempts=5)`: 1 request lấy n completions (share prefill — rẻ hơn n request); retry exponential backoff + jitter cho lỗi transient (connection/timeout/429/5xx); **không bao giờ trả `""` im lặng** — raise để caller re-queue.
- `PredictionCache` (JSONL, append-từng-dòng crash-safe): key `(qid, strategy_id, sample_index)`, có `status: ok|failed|repaired|repair_failed`. Chạy lại lệnh = tự resume từ cache; `--retry-failed` để re-queue câu hỏng.
- Scheduler `as_completed`: kết quả nào xong ghi ngay xuống đĩa, crash chỉ mất request đang bay.
- Ops: set `min_replicas = 1` trước run dài (tránh cold start giữa chừng), trả về 0 sau; client concurrency 24 (< replica_concurrency 32 của server).

### 3.2 Decoding chống runaway

Mỗi (câu, strategy) = 2 groups:
- **Greedy**: n=1, temp=0, `repetition_penalty=1.05`, max_new_tokens=3072.
- **Sampled**: n=7, temp=0.6, top_p=0.95, top_k=20, `repetition_penalty=1.05`, max_new_tokens=2048 (p99 output khỏe mạnh chỉ ~1500).

→ Full dev = 584 × 2 strategies × 2 groups = **2,336 requests** (không phải 9,344).

### 3.3 Extraction v2 + Repair

Ladder: (1) rỗng → repair; (2) không có `</think>` → tag runaway, **không mót fragment**, đi thẳng repair; (3) **JSON sau `</think>` cuối cùng** → `"Program syntax"` (+ đọc `"Numerical result"` làm tín hiệu phụ); (4) JSON bất kỳ (khối cuối); (5) quét dòng từ dưới lên, chỉ nhận dòng "thuần program". Hậu xử lý: dedupe step trùng, validate #refs, reject prose.

**Repair pass**: câu nào không có program hợp lệ / execute crash → 1 lần re-ask guided JSON `{"Program syntax": str}` (kèm câu hỏi + context + 1200 ký tự cuối bản nháp lỗi + luật DSL, 256 tokens) → re-execute.

### 3.4 Evaluator v2 (công cụ của MÌNH — sửa là ăn điểm trực tiếp)

- `flatten_nested_program`: `divide(subtract(500,400),400)` → `subtract(500, 400), divide(#0, 400)`.
- `fuzzy_row_match`: exact → substring → bỏ dấu/hoa-thường/đuôi đơn vị "(x)" → Jaccard token ≥ 0.6.
- `extract_numbers_v2`: `"$ -61.1 ( 61.1 )"` → `[-61.1]` (bug này làm chính doc-test của evaluator.py sai — fix là khôi phục hành vi đúng dự kiến).

### 3.5 Voting (self-consistency + ensemble)

16 candidates/câu (2 strategies × 8 samples) → execute hết → **cluster theo giá trị** với tolerance `max(1e-4, 0.5%·|v|)` → vote có trọng số (greedy +0.5; candidate có executed value khớp "Numerical result" tự khai +0.25) → nhận nếu `confidence ≥ 0.5 & members ≥ 3 & margin ≥ 0.15`, ngược lại đánh dấu `needs_judge`. Giá trị đại diện = greedy member nếu có trong cluster thắng.

### 3.6 Retrieval few-shot (strategy B của ensemble)

TF-IDF char n-gram (2,4) trên 2986 câu train (build vài giây, không torch) → mỗi câu test/dev lấy top-8, chọn 3 shot đa dạng op-signature làm few-shot **per-question** (gold flat program → dạy đúng format executable). **Nền template = s4fix** (CÙNG base với strategy A — chỉ khác nguồn few-shot: A dùng few-shot cố định đã sửa, B thay bằng retrieval per-question). Ensemble vì thế chỉ đổi MỘT biến (nguồn shot) → dễ quy kết nguyên nhân và loại bias của việc trộn 2 base khác nhau. **Gate trên dev**: nếu retrieval-alone < s4fix − 2pp → fallback strategy B = s4fix few-shot cố định (ensemble suy biến thành 2× cùng template nhưng khác seed sampling, vẫn có lợi cho self-consistency).

### 3.7 LLM Judge (DeepSeek-R1-Distill-Qwen-7B, self-host trên Cerebrium)

> **⚠️ OPTIONAL / STRETCH — NGOÀI CRITICAL PATH.** Submission mặc định = kết quả vote+ensemble (A4), **KHÔNG phụ thuộc tầng judge**. Chỉ chạy P7 khi HỘI ĐỦ cả 3: (a) đã có `submission.csv` an toàn trên đĩa từ A4; (b) còn ≥3h GPU + đủ thời gian trước deadline 2026-07-09; (c) A5 trên dev thực sự thắng A4. Rủi ro cao (2× deploy swap ~20', tải 15GB weights, có thể làm bẩn git state của `cerebrium.toml`) mà lợi kỳ vọng chỉ +1-3pp → **mặc định BỎ QUA nếu deadline sát**. Phần dưới là thiết kế chi tiết để dùng khi và chỉ khi quyết định bật.

**Model & hạ tầng (1 app Cerebrium duy nhất — swap model tuần tự):**
- Judge = `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7B < cap 9B của luật Kaggle — model open, chỉ cần khai báo trong report). 7B bf16 ≈ 15GB → chạy MỘT MÌNH vẫn vừa A10 24GB (`--max-model-len 16384 --gpu-memory-utilization 0.92`), nhưng KHÔNG thể chạy song song với Qwen3.5 trên cùng GPU.
- **Cách triển khai với đúng 1 app**: tầng judge chỉ chạy SAU khi toàn bộ generation + repair đã nằm an toàn trong cache JSONL → không cần 2 model cùng lúc. Quy trình swap:
  1. Xong toàn bộ việc cần Qwen (dev A1-A4 + test generate/repair) → sửa model trong `cerebrium.toml` entrypoint thành R1-7B (+ gpu-mem-util 0.92) → `cerebrium deploy` (~5-10 phút; weights 15GB tải 1 lần rồi cache persistent storage).
  2. Chạy judge cho **cả dev (ablation A5) lẫn test trong cùng cửa sổ này** → chỉ cần **2 lần swap tổng cộng** (sang R1 rồi về Qwen), không phải 4.
  3. Swap về config Qwen gốc → `cerebrium deploy` → `cerebrium.toml` trở lại nguyên trạng (git diff sạch).
- Guard an toàn: `judge.py` trước khi chạy PHẢI gọi `models.list()` assert model đang serve == `JUDGE_MODEL` (tránh chấm nhầm bằng Qwen); tương tự `predict.py generate` assert đang serve Qwen.
- `.env`: `JUDGE_BASE_URL` (= chính `CEREBRIUM_BASE_URL`), `JUDGE_API_KEY` (= `CEREBRIUM_API_KEY`), `JUDGE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `JUDGE_MAX_CALLS=300`. Thiếu config / model chưa swap → pipeline tự fallback majority-vote, không chết.

**Nguyên tắc prompt — KHÔNG chặn thẻ `<think>` (quan trọng):**
- R1-Distill là reasoning model: **phải để nó suy nghĩ tự do trong `<think>...</think>` trước khi chấm**. Tuyệt đối KHÔNG dùng guided_json/ép JSON thuần ngay từ token đầu — làm vậy bóp chết chuỗi suy nghĩ, độ chính xác logic giảm 30-40%.
- Verdict đặt **ngay sau `</think>`** ở định dạng text đơn giản, bóc bằng **regex**:
  ```
  SELECTION: <số cluster 1-based, hoặc -1 nếu tất cả sai>
  PROGRAM: <chương trình DSL phẳng đã sửa, hoặc NONE>
  FINAL_VALUE: <số>
  ```
  Regex: `SELECTION:\s*(-?\d+)`, `PROGRAM:\s*(.+)`, `FINAL_VALUE:\s*(-?[\d.,eE+-]+)` — áp dụng trên phần text SAU `</think>` cuối cùng. Parse fail → 1 lần re-ask, fail nữa → fallback majority.
- Sampling theo khuyến nghị DeepSeek cho R1: `temperature=0.6, top_p=0.95`, **không system prompt** (toàn bộ chỉ dẫn nằm trong user message), `max_new_tokens=4096` (thinking dài), không repetition_penalty ép.

**Vận hành:**
- Chỉ gọi cho câu `needs_judge` (~100-200 câu), cap `JUDGE_MAX_CALLS`, verdict cache vào `runs/phase3/judge_cache.jsonl` (key qid + config hash — chạy lại ablation không tốn GPU gọi lại; lưu cả phần `<think>` để debug).
- Input: câu hỏi + context/bảng + luật DSL + top-3 cluster (value, votes, program đại diện) + yêu cầu tự tính lại từng ứng viên, kiểm tra dấu/chiều và scale tỷ lệ thập phân.
- **Thứ tự tin cậy verdict**: PROGRAM execute local OK (bằng evaluate_program_v2) → dùng giá trị execute; else SELECTION hợp lệ → giá trị đại diện cluster đó; else FINAL_VALUE hữu hạn; else majority vote. So sánh 2 mode trên dev (select-only vs select+author), giữ mode thắng.
- Chi phí: không tốn API ngoài — chỉ thêm ~0.5-1.5h GPU A10 trên chính app hiện có cho ~300-500 call (dev + test) × thinking dài, cộng ~2×10 phút cho 2 lần deploy swap.

### 3.8 Submission formatting + sanity gate

- **KHÔNG tự viết CSV.** `predict.py submit` chỉ tổng hợp kết quả cuối rồi xuất `runs/phase3/test_predictions.json` (`{id: value}`), sau đó **BẮT BUỘC** đi qua **`format_submission.py`** (có sẵn, dùng nguyên trạng) để sinh `submission.csv`. File này đảm bảo đúng schema Kaggle **3 cột `id,Usage,predicted_value`** (Usage=`"Public"`), đúng thứ tự hàng theo `data/test.json`, tự điền `0.0` cho câu thiếu/rỗng/null/NaN/non-numeric. → Xóa nguy cơ **mất cột `Usage`** (vi phạm rule 10 submission validity) và bỏ luôn redundancy tự viết CSV.
- `predict.py submit` chạy được trên `test_cache.jsonl` ở **bất kỳ trạng thái nào** (partial cũng OK — câu chưa có thì `format_submission.py` tự điền `0.0`) → luôn tồn tại **1 `submission.csv` hợp lệ trên đĩa** làm phao cứu sinh chống crash/hết giờ.
- `predict.py check` (bắt buộc pass **trước khi upload Kaggle**) chạy trên cặp `test_predictions.json` + `submission.csv` vừa sinh: xác nhận CSV **đúng 3 cột `id,Usage,predicted_value`**; đủ 494 id đúng thứ tự; **số câu 0.0 ≤ 10** (baseline có 102!); phân bố value tương đồng dev; ≥485/494 câu có candidate hợp lệ; runaway <5%.
- ⚠️ Lưu ý format số: `format_submission.py` ghi `str(float)` → giá trị rất nhỏ có thể ra **scientific notation** (`2.5e-05`) và `-0.0` giữ nguyên (file này không được sửa). Vì vậy `check` quét CSV tìm `e/E`-notation và `-0.0`; nếu có → chạy bước **fixup render số** (viết lại đúng các ô đó dạng plain decimal, giá trị và 3 cột giữ nguyên) rồi check lại.

### 3.9 Logging & artifacts (lưu log + kết quả như giai đoạn 1)

Mô phỏng đúng phong cách `runs/exp_self/` của giai đoạn 1 — mọi run đều để lại dấu vết đầy đủ, tự động, trong `runs/phase3/`:

```
runs/phase3/
├── run.log                        # log toàn phiên (như exp_self/run.log): logging → FileHandler + console,
│                                  #   timestamp, mỗi request retry/fail, tiến độ, token usage định kỳ
├── phase3_config.json             # snapshot toàn bộ tham số mỗi lần chạy (như args.json): strategies, K,
│                                  #   decoding params, vote thresholds, judge model/mode, commit hash, thời điểm
├── strategy_s4fix.json            # strategy A (clone iter_004 đã sửa)
├── strategy_retr_base.json        # strategy B = nền s4fix + retrieval few-shot per-question (chỉ khác A ở nguồn shot)
├── dev_cache.jsonl                # cache generation per-sample (resume) — dev
├── test_cache.jsonl               # cache generation per-sample (resume) — test
├── judge_cache.jsonl              # verdict judge per-question (kèm prompt hash, model, usage tokens)
├── ablation_A1_eval.json          # kết quả chấm per-question CÙNG SCHEMA với iter_XXX_eval_dev.json
├── ablation_A3_eval.json          #   (question, gold_answer, gold_val, predicted_answer, predicted_val,
├── ablation_A4_eval.json          #    is_correct, question_type, raw_output/vote metadata)
├── ablation_A5_eval.json
├── ablation_report.json           # bảng tổng hợp: accuracy từng ablation, accuracy theo question_type,
│                                  #   runaway rate, repair rate, judge stats, token cost — "learning curve" của giai đoạn 2
├── test_details.json              # như submission_details.json cũ nhưng giàu hơn: per-question toàn bộ
│                                  #   candidates, cluster, vote outcome, judge verdict, giá trị cuối
└── token_usage.json               # tổng kết TokenBudget (qwen in/out + judge in/out) cho report THINKFLIC
```

Nguyên tắc:
- **Mỗi lệnh `predict.py *` tự ghi** vào `run.log` (mode append, có tag subcommand) + cập nhật `phase3_config.json` khi tham số đổi — không cần nhớ bật gì.
- **`ablation_AX_eval.json` dùng đúng schema `iter_XXX_eval_dev.json`** của giai đoạn 1 → tái dùng được các công cụ phân tích sẵn có (`src/analysis.py` failure_mode_report) và dễ so sánh trước/sau.
- **Token tracking**: tái dùng `TokenBudget` (executor.py) cộng dồn từ usage của từng response (Qwen) + usage judge → xuất `token_usage.json` (yêu cầu report THINKFLIC: compute + cost).
- `submission.csv` + `test_details.json` là cặp bằng chứng cuối (như `submission.csv` + `submission_details.json` giai đoạn 1).

---

## 4. Lộ trình thực thi + ablation (dừng ở đâu vẫn submit được ở đó)

| Phase | Việc | GPU | Gate/Kỳ vọng dev-584 |
|-------|------|-----|----------------------|
| P0 | Cài scikit-learn; (chỉ khi làm P7) thêm JUDGE_* vào .env; strategy_s4fix.json (sửa few-shot #4 + 2 hint: chiều thay đổi B−A, số kiểu Việt 11.228→11228); khởi tạo runs/phase3/ + logging | 0 | — |
| P1 | Extraction v2 + Evaluator v2 + **rescore offline** trên raw_output CŨ | **0** | **A0: 0.579 → ~0.63** (gate ≥ +3pp mới đi tiếp) |
| P2 | complete_many + cache/resume + smoke 16 câu (assert n>1 hoạt động, kill/resume OK) | ~0 | runaway < 5% |
| P3 | Greedy full dev + repair | ~1h | **A1/A2: ~0.66-0.71** |
| P4 | Self-consistency K=8 + voting | ~1h | **A3: +3-6pp** |
| P5 | Retrieval strategy + ensemble 16 candidates | ~1.5h | **A4: +1-4pp** |
| P6 | Freeze config generation (từ A1-A4) → **test generate + repair** (vẫn Qwen) — mọi thứ vào test_cache.jsonl. **Ngay khi xong: chạy `submit` → có submission.csv A4 an toàn trên đĩa** | ~2-3h | — |
| **P7 (OPTIONAL)** | Chỉ khi đủ điều kiện §3.7: **Swap model → R1-7B** (sửa toml + deploy) → judge dev (A5: chọn mode/threshold) + judge test trong CÙNG cửa sổ → **swap về Qwen** (toml về nguyên trạng). **Bỏ qua nếu deadline sát** | ~0.5-1.5h | **A5: +1-3pp** |
| P8 | Tổng hợp vote (**+judge nếu có P7**) → `submit` (JSON) → `format_submission.py` → `check` → submission.csv → bạn upload Kaggle | 0 | mục tiêu **≥0.78** |
| P9 | Re-run 6 graders + ghi chú tái lập cho report | 0 | 6/6 PASS |

**Critical path = P0-P6 + P8** (đã đủ để submit ≥0.78 mục tiêu). **P7 là nhánh optional**, P8 không phụ thuộc P7 — nếu bỏ P7 thì submission = kết quả A4.

**Tổng chi phí ước tính (critical path P0-P6+P8): ~4-6h GPU A10 (~$5-9 Cerebrium).** P7 optional cộng thêm ~0.5-1.5h GPU + 2 lần deploy — chỉ tốn khi bật.

Ngân sách điểm (cộng dồn kỳ vọng): fix runaway ~+12pp → extraction/crash ~+10pp (chồng lấn) → SC vote +4-8pp → ensemble +1-4pp ⇒ vùng hạ cánh **0.74-0.80 KHÔNG cần judge**. Judge (P7 optional) chỉ là lớp phủ +1-3pp trên nền đó.

### Lệnh chạy (từ `assignment03`, son_env)

```powershell
# P1 — không tốn GPU
python predict.py rescore --source runs\exp_self\iter_004_eval_dev.json

# P3 — dev greedy + repair
python predict.py generate --split validation --strategies s4fix --groups greedy --cache runs\phase3\dev_cache.jsonl
python predict.py repair   --split validation --cache runs\phase3\dev_cache.jsonl
python predict.py score    --split validation --ablation A1

# P4/P5 — thêm sampled + retrieval (cache tự reuse phần đã chạy)
python predict.py generate --split validation --strategies s4fix --groups sampled --k 7 --cache runs\phase3\dev_cache.jsonl
python predict.py generate --split validation --strategies retr  --groups greedy,sampled --k 7 --cache runs\phase3\dev_cache.jsonl
python predict.py score    --split validation --ablation A4

# P6 — test generate + repair (vẫn đang serve Qwen)
python predict.py generate --split test --strategies s4fix,retr --groups greedy,sampled --k 7 --cache runs\phase3\test_cache.jsonl
python predict.py repair   --split test --cache runs\phase3\test_cache.jsonl
# Ngay khi xong P6: sinh submission A4 an toàn trên đĩa (phao cứu sinh — CHƯA upload)
python predict.py submit --cache runs\phase3\test_cache.jsonl --out runs\phase3\test_predictions.json
python format_submission.py --predictions runs\phase3\test_predictions.json --output-file submission.csv

# P7 — (OPTIONAL) chỉ khi đủ điều kiện §3.7. Swap model sang R1 judge (sửa cerebrium.toml -> cerebrium deploy), rồi:
python predict.py judge --split validation --cache runs\phase3\dev_cache.jsonl    # A5 trên dev
python predict.py score --split validation --ablation A5
python predict.py judge --split test --cache runs\phase3\test_cache.jsonl
# swap toml về Qwen -> cerebrium deploy (toml trở lại nguyên trạng)

# P8 — tổng hợp + kiểm tra + xuất file (KHÔNG tự viết CSV — đi qua format_submission.py)
python predict.py submit --cache runs\phase3\test_cache.jsonl --out runs\phase3\test_predictions.json
python format_submission.py --predictions runs\phase3\test_predictions.json --output-file submission.csv
python predict.py check  --submission submission.csv
```

---

## 5. Rủi ro & phòng bị

| Rủi ro | Phòng bị |
|--------|----------|
| Pod restart giữa run (nguồn gốc 46 câu rỗng) | Retry backoff + JSONL resume + min_replicas=1 khi chạy dài |
| temp 0.6 vẫn loop | rep_penalty 1.05 + cap 2048 tokens + monitor runaway ở smoke; knob dự phòng presence_penalty |
| `n>1` quirk trên vLLM 0.24 deployment | Smoke assert len(choices)==n trước run lớn; fallback n request tuần tự |
| Judge override câu đúng | Đo trên dev (A5); execution-first ladder; mode select-only; cap calls; fallback majority |
| Swap model trục trặc sát deadline (deploy fail / R1 không load) | **P7 là OPTIONAL — bỏ hẳn nếu rủi ro.** submission = A4 đã có sẵn trên đĩa từ cuối P6, KHÔNG phụ thuộc judge; nếu vẫn muốn thử: weights R1 tải trước lúc rảnh |
| Chạy nhầm model (generate bằng R1 / judge bằng Qwen) | Guard assert `models.list()` == model kỳ vọng trong cả `generate` lẫn `judge` trước khi bắn request |
| Grader hỏng do sửa code | Chỉ add-only vào call-path được test; re-run 6 graders ở P9 |
| Metric Kaggle lạ (rounding/format) | Plain decimal formatter, không blank, id khớp tuyệt đối |
| Cạn thời gian (deadline 9/7) | Thứ tự phase = thứ tự giá trị; dừng sau P3 vẫn ~0.66-0.71 ≫ 0.599 |

---

## 6. Việc bạn cần chuẩn bị

1. **(CHỈ KHI QUYẾT ĐỊNH LÀM P7 OPTIONAL) Judge = DeepSeek-R1-Distill-Qwen-7B trên chính app Cerebrium hiện có** (swap model tuần tự ở P7, chỉ 2 lần deploy: sang R1 rồi về Qwen — `cerebrium.toml` cuối cùng trở lại nguyên trạng). Không cần app thứ hai, không cần API ngoài. **Nếu bỏ P7 thì mục này không cần chuẩn bị gì.**
2. Set `min_replicas = 1` trên Cerebrium trước các run dài (P3-P7), trả về 0 sau khi xong.
3. Sau P8: upload `submission.csv` lên Kaggle (1 lần duy nhất). Toàn bộ bằng chứng cho report THINKFLIC nằm sẵn trong `runs/phase3/` (run.log, ablation_report.json, token_usage.json, test_details.json).
