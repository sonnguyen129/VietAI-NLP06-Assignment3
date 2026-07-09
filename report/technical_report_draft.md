# Technical Report — Assignment 03: Self-Improving AI for Financial Question Answering

> **DRAFT** — các ô 【Điền: ...】 cần hoàn thiện trước khi convert sang `report.pdf`.
> Độ dài khuyến nghị 4-6 trang (không tính phụ lục).

---

## 1. Thông tin cá nhân

| Mục | Thông tin |
|---|---|
| Họ tên | 【Điền: họ tên đầy đủ】 |
| MSSV | 【Điền: mã số sinh viên】 |
| Lớp | 【Điền: lớp / khóa NLP06】 |
| Kaggle username | 【Điền: username Kaggle】 |
| Kaggle team name | 【Điền: tên team trên Kaggle (cá nhân)】 |
| Hình thức nộp | Cá nhân (100% khối lượng công việc) |
| Video thuyết trình | 【Điền: link Google Drive — "Anyone with the link can view"】 |

---

## 2. Tóm tắt & Kết quả Kaggle

Hệ thống EvoAgent giải bài toán hỏi–đáp tài chính tiếng Việt (FinQA-style): đọc văn bản + bảng số liệu, sinh chương trình DSL phẳng (`subtract(a,b), divide(#0,b)`...), thực thi chương trình để ra giá trị số cuối cùng.

**Kết quả chính:**

| Mốc | Dev accuracy | Kaggle Public LB |
|---|---|---|
| Baseline (EvoAgent thuần, iter_004, greedy) | 0.5792 (dev-240) | **0.59919** |
| Hệ thống cuối (Phase 3) | **0.7346** (full dev-584) | **0.77935** |
| Cải thiện | +15.5 pp | **+18.0 pp** |

Submission cuối: 【Điền: tên submission trên Kaggle】 · Rank tại thời điểm nộp: 【Điền: rank/tổng số team】 (top LB lúc đó: 0.80971).

**Ba nguồn cải thiện chính** (chi tiết ở mục 5): (1) vá các lỗi hệ thống của baseline — request thất bại bị nuốt thành 0.0, chuỗi suy nghĩ chạy vô hạn, extraction gom rác, few-shot dạy sai cú pháp (~+6 pp); (2) self-consistency voting trên 16 candidates/câu (~+7 pp); (3) ensemble với retrieval few-shot per-question (~+2.4 pp).

---

## 3. Pipeline cuối cùng

### 3.1 Kiến trúc tổng thể

```
┌── Máy local (orchestration) ──────────────────────────────────┐
│ predict.py: generate → repair → vote → submit → check          │
│  • Cache JSONL per-sample (resume được, crash-safe)            │
│  • Retry exponential backoff cho mọi request                   │
│         │ HTTP (OpenAI-compatible /v1/completions, n>1)        │
└─────────┼──────────────────────────────────────────────────────┘
          ▼
┌── Cerebrium (1 app, GPU A10 24GB) ────────────────────────────┐
│ vLLM 0.24.0 serve QuantTrio/Qwen3.5-4B-AWQ (~5.6GB AWQ)       │
│ max-model-len 16384 · scale-to-zero khi rảnh                   │
└────────────────────────────────────────────────────────────────┘
```

Mỗi câu hỏi test đi qua: **2 strategies × (1 greedy + 7 sampled) = 16 candidates** → extraction → thực thi chương trình → clustering giá trị → weighted vote → giá trị cuối.

### 3.2 Các khối chính

**(a) Prompt strategies (2 biến thể, cùng base template).** Strategy A (`s4fix`) kế thừa strategy tốt nhất của vòng tiến hóa EvoAgent (iter_004) với 2 sửa đổi: few-shot #4 vốn dạy nested call `divide(subtract(500,400),400)` (chính evaluator của đề không chạy được) được đổi thành dạng phẳng `subtract(500,400), divide(#0,400)`; bổ sung 2 hint — quy tắc chiều thay đổi ("từ A đến B = B − A, kết quả có thể âm") và quy tắc số kiểu Việt ("11.228 tỷ" = 11228). Strategy B (`retr`) dùng **cùng template** nhưng thay few-shot cố định bằng **retrieval per-question**: TF-IDF char n-gram (2,4) trên 2,986 câu train, lấy top-8 rồi chọn 3 shot đa dạng op-signature, few-shot chính là gold program phẳng của train — dạy đúng format thực thi được. Ensemble chỉ thay đổi đúng một biến (nguồn few-shot) nên dễ quy kết nguyên nhân.

**(b) Decoding chống runaway.** Baseline chạy greedy temp=0 không repetition penalty → 21% output rơi vào vòng lặp suy nghĩ vô hạn (không bao giờ đóng `</think>`), nhóm này đúng **0%**. Pipeline cuối: greedy có `repetition_penalty=1.05` (cap 3072 tokens) + 7 sampled `temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.05` (cap 2048) sinh trong **một request `n=7`** (chia sẻ prefill trên vLLM — chi phí gần bằng 1 request).

**(c) Extraction v2.** Baseline `extract_answer` gom mọi mảnh `op(...)` trong toàn bộ monologue thành một "chương trình" → 47/240 câu dev crash evaluator. Extraction v2 dùng ladder ưu tiên: JSON sau `</think>` cuối → JSON bị cắt cụt (regex key `"Program syntax"`) → quét dòng thuần-program từ dưới lên; **tuyệt đối không mót fragment từ output runaway** — nhóm đó chuyển cho repair. Kèm validate cú pháp (op hợp lệ, #ref đúng thứ tự, ≤8 bước, dedupe bước lặp).

**(d) Repair pass.** Mọi sample không có chương trình thực thi được → 1 lần re-ask guided-JSON (`{"Program syntax": str}`, 256 tokens, kèm 1200 ký tự cuối bản nháp lỗi). Trên dev: vá thành công 660/987 sample hỏng trong 6.5 phút.

**(e) Evaluator v2 (thực thi chương trình).** Ba sửa đổi so với evaluator gốc: tự động **flatten nested call** thành SSA phẳng; **fuzzy row match** cho table ops (bỏ dấu/hoa-thường/hậu tố đơn vị "(x)", Jaccard ≥ 0.6) — sửa lỗi "P/E" không khớp "P/E (x)"; **chuẩn hóa số âm kế toán** `"$ -61.1 ( 61.1 )"` → −61.1 (bug gốc khiến `table_max` chọn sai dấu — chính doc-test của `evaluator.py` cũng bị sai vì bug này).

**(f) Self-consistency voting.** 16 candidates được thực thi hết, **vote theo GIÁ TRỊ đã execute** (không vote theo text): cluster single-link với tolerance `max(1e-4, 0.5%·|v|)`; trọng số 1.0 + 0.5 nếu là greedy + 0.25 nếu giá trị execute khớp "Numerical result" model tự khai; giá trị cuối = greedy member trong cluster thắng (nếu có). Ngưỡng đồng thuận: confidence ≥ 0.5, ≥ 3 thành viên, margin ≥ 0.15.

**(g) Reliability layer.** Điều tra baseline phát hiện **46/494 câu test (9.3%) có raw output RỖNG** — request fail khi pod restart bị nuốt im lặng thành 0.0. Pipeline cuối: retry exponential backoff + jitter (phân loại lỗi transient 429/5xx/timeout vs permanent 4xx), scheduler `as_completed` ghi cache JSONL từng kết quả ngay khi xong (key `(qid, strategy, sample_index)` + status), chạy lại lệnh tự resume. Kết quả: **0/4,672 request thất bại** trên dev, 0 câu bị 0.0 oan trên test.

**(h) Post-processing & submission gate.** Giá trị cuối qua `format_submission.py` (nguyên trạng của đề); gate `predict.py check` xác nhận: đúng 3 cột, đủ 494 id đúng thứ tự, số câu 0.0 ≤ 10 (baseline có 102!), không scientific notation/-0.0, phân bố giá trị tương đồng dev.

### 3.3 Những gì KHÔNG dùng trong hệ thống cuối

- **Fine-tuning**: không dùng (giữ nguyên serving config; xem mục 7).
- **LLM-as-judge**: đã thử nghiệm đầy đủ với DeepSeek-R1-Distill-Qwen-7B nhưng **loại khỏi pipeline cuối** vì ablation trên dev cho kết quả âm (mục 5.3).

---

## 4. Dữ liệu ngoài & dữ liệu synthetic

- **Dữ liệu ngoài: KHÔNG sử dụng.** Toàn bộ few-shot (cả cố định lẫn retrieval) lấy từ `data/train.json` được cung cấp; không crawl, không dataset bổ sung, không tra cứu/tái tạo nhãn test dưới bất kỳ hình thức nào.
- **Dữ liệu synthetic**: các chuỗi reasoning CoT trong few-shot của strategy s4fix do chính pipeline EvoAgent (Qwen3.5-4B) sinh ra ở giai đoạn 1 và được verify bằng cách thực thi chương trình (chỉ giữ khi kết quả khớp gold). Không dùng model ngoài để sinh nhãn hay pseudo-label.
- **Model sử dụng**: `QuantTrio/Qwen3.5-4B-AWQ` (4B, model chính — toàn bộ generation/repair) và `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7B, self-host thử nghiệm judge — không có trong pipeline cuối). Cả hai ≤ 9B, tuân thủ luật cuộc thi; không dùng API model đóng nào.

---

## 5. Thí nghiệm, Ablation & Lựa chọn hệ thống cuối

### 5.1 Điều tra lỗi baseline (động lực thiết kế)

Phân tích per-question trên artifacts của baseline (dev-240 + 494 raw output test) xác định điểm rơi:

| Lỗi hệ thống | Bằng chứng | Tỷ trọng |
|---|---|---|
| Runaway CoT (không đóng `</think>`) | 51/240 dev → **0% đúng** | ~50% tổng lỗi dev |
| Request fail bị nuốt thành 0.0 | 46/494 test raw output rỗng | 9.3% test |
| Extraction gom rác → crash evaluator | 47/240 dev | ~46% lỗi dev |
| Few-shot #4 dạy nested call | 28/494 test program nested → 0.0 | 5.7% test |
| Lỗi dấu/chiều thay đổi | 7/240 dev có pred = −gold | ~3% |

Phát hiện then chốt: trong các output đóng `</think>` tử tế, **100% chứa khối JSON parse được** → chỉ cần diệt runaway + extraction đúng chỗ là giải quyết phần lớn.

### 5.2 Bảng ablation chính (full dev-584, scorer abs ≤ 1e-4)

| # | Cấu hình (cộng dồn) | Dev acc | Δ |
|---|---|---|---|
| — | Baseline iter_004 (dev-240, tham chiếu) | 0.5792 | — |
| A1 | Greedy + toàn bộ fixes (decoding, extraction v2, evaluator v2, s4fix, repair) | **0.6404** | +6.1 pp |
| A3 | + Self-consistency K=8, vote theo giá trị | **0.7106** | +7.0 pp |
| A4 | + Ensemble strategy retrieval (16 candidates) — **HỆ THỐNG CUỐI** | **0.7346** | +2.4 pp |
| A5 | + LLM-judge R1-Distill-7B cho câu bất đồng thuận | 0.7295 | **−0.5 pp** (loại) |

Accuracy theo loại câu hỏi (A4 vs baseline): `table_op` 0.33 → **0.71**; `division` 0.68 → 0.79; `addition` 0.39 → 0.65 (vẫn yếu nhất); `subtraction` 0.62 → 0.73; `multiplication` 0.60 → 0.80.

### 5.3 Ablation âm: LLM-as-judge (bằng chứng phương pháp luận)

Thử nghiệm: các câu vote không đạt ngưỡng đồng thuận (~20-35%) được đưa cho DeepSeek-R1-Distill-Qwen-7B (self-host trên cùng app Cerebrium, swap model tuần tự) phán xử. Thiết kế tôn trọng đặc thù reasoning model: **không chặn thẻ `<think>`** (không ép JSON từ token đầu — làm vậy giảm mạnh chất lượng suy luận), verdict bóc bằng regex sau `</think>` (`SELECTION/PROGRAM/FINAL_VALUE`), và chỉ tin chương trình judge viết lại **sau khi thực thi local thành công**. Kết quả trên dev: judge override 33 câu, làm **giảm** 3 câu ròng (0.7346 → 0.7295) — model 7B không đủ tin cậy để thắng majority vote của 16 candidates trên miền tài chính tiếng Việt. **Quyết định: loại judge khỏi hệ thống cuối, giữ A4.** Toàn bộ chuỗi suy nghĩ của judge được lưu tại `runs/phase3/judge_cache.jsonl` làm bằng chứng.

### 5.4 Phương pháp validate

Toàn bộ quyết định dựa trên **full dev 584** (baseline chỉ dùng slice 240); test không có nhãn nên tuyệt đối không tune trên test; chỉ submit Kaggle 2 lần (baseline + final), không probe leaderboard.

---

## 6. Tái lập (Reproduction)

### 6.1 Môi trường & dependencies

- Python 3.13 (Anaconda env), các gói chính: `openai`, `transformers` (tokenizer-only), `datasets`, `scikit-learn`, `pydantic`, `python-dotenv`, `numpy`, `tqdm` (xem `requirements.txt`).
- Server: Cerebrium 1 app A10 — `cerebrium.toml` (vLLM `0.24.0` cài qua pip, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--max-model-len 16384 --gpu-memory-utilization 0.90`).
- `.env`: `CEREBRIUM_BASE_URL`, `CEREBRIUM_API_KEY`, `CEREBRIUM_MODEL=QuantTrio/Qwen3.5-4B-AWQ`, `HF_TOKEN`.

### 6.2 Lệnh tái lập đầy đủ

```powershell
# Dev (validate):
python predict.py generate --split validation --strategies s4fix --groups greedy,sampled --k 7 --cache runs\phase3\dev_cache.jsonl --concurrency 24
python predict.py repair   --split validation --cache runs\phase3\dev_cache.jsonl
python predict.py generate --split validation --strategies retr --groups greedy,sampled --k 7 --cache runs\phase3\dev_cache.jsonl --concurrency 24
python predict.py repair   --split validation --cache runs\phase3\dev_cache.jsonl
python predict.py score --ablation A1 ; python predict.py score --ablation A3 ; python predict.py score --ablation A4

# Test + submission:
python predict.py generate --split test --strategies s4fix,retr --groups greedy,sampled --k 7 --cache runs\phase3\test_cache.jsonl --concurrency 24
python predict.py repair   --split test --cache runs\phase3\test_cache.jsonl
python predict.py submit --ablation A4 --cache runs\phase3\test_cache.jsonl --out runs\phase3\test_predictions.json
python format_submission.py --predictions runs\phase3\test_predictions.json --output-file submission.csv
python predict.py check --submission submission.csv
```

Mọi lệnh idempotent (cache JSONL tự resume); log tự ghi vào `runs/phase3/run.log`, tham số snapshot vào `phase3_config.json`.

### 6.3 Cấu hình model & sampling

| Tham số | Greedy | Sampled (n=7) | Repair | Judge (thử nghiệm) |
|---|---|---|---|---|
| temperature | 0.0 | 0.6 | 0.0 | 0.6 |
| top_p / top_k | 1.0 / — | 0.95 / 20 | 1.0 / — | 0.95 / — |
| repetition_penalty | 1.05 | 1.05 | 1.0 | 1.0 |
| max_new_tokens | 3072 | 2048 | 256 (guided JSON) | 4096 |

Vote: tolerance `max(1e-4, 0.5%)`, ngưỡng chấp nhận confidence 0.5 / members 3 / margin 0.15.

### 6.4 Compute, chi phí & commit

| Hạng mục | Input tokens | Output tokens |
|---|---|---|
| Generation (dev + test, Qwen) | 55,941,840 | 17,239,308 |
| Repair (Qwen, guided) | 8,792,622 | 170,416 |
| Judge thử nghiệm (R1-7B) | 118,240 | 41,844 |
| **Tổng** | **~64.9M** | **~17.5M** |

- GPU: ~14-15 giờ A10 (Cerebrium, scale-to-zero khi rảnh). Chi phí thực tế: 【Điền: tổng $ từ Cerebrium Dashboard → Billing, ước tính ~$15-20】.
- Không phát sinh chi phí API ngoài (judge tự host).
- Commit hash: 【Điền: hash sau commit cuối — tại thời điểm draft là `6df8af0657b1c629013954bb80a133d2a0a3318f`, cần commit nốt các thay đổi giai đoạn 2 rồi cập nhật】.

---

## 7. Hạn chế & Bài học

**Hạn chế:**
1. **Runaway CoT chưa trị tận gốc** — repetition penalty 1.05 giảm nhưng vẫn ~19.5% sample runaway; hệ thống bù bằng vote + repair thay vì sửa tận gốc (cần fine-tune hoặc model mới hơn).
2. **`addition` vẫn yếu nhất (0.65)** — thực chất là nhóm câu đa bước phức tạp bị classifier gán nhãn addition; cần phân loại câu hỏi tốt hơn để nhắm few-shot chính xác.
3. **Dev–LB lệch +4.5pp** (0.735 vs 0.779) — scorer nội bộ abs ≤ 1e-4 khắt khe hơn metric Kaggle; may mắn lệch theo hướng có lợi nhưng làm giảm độ chính xác của dự báo dev.
4. **Judge 7B thất bại** — với budget model nhỏ, verifier không thắng được self-consistency; muốn judge hiệu quả cần model lớn hơn đáng kể hoặc verifier được huấn luyện chuyên biệt.
5. Chưa thử fine-tuning (ràng buộc giữ nguyên serving + thời gian) — hướng tiềm năng nhất còn lại.

**Bài học lớn nhất:** phần lớn điểm cải thiện (+11 trong +18 pp) đến từ **kỹ nghệ hệ thống** — điều tra per-question để tìm đúng chỗ rơi điểm (request bị nuốt, extraction gom rác, few-shot dạy sai, decoding sai chế độ) — chứ không phải từ model thông minh hơn. "Đo trước, sửa đúng chỗ, mỗi thay đổi một ablation" hiệu quả hơn nhiều so với đổi kiến trúc theo cảm tính; và một ablation âm được đo nghiêm túc (judge) cũng giá trị không kém ablation dương.

---

## Phụ lục A — Bằng chứng đính kèm gói ThinkFlic

| File | Nguồn |
|---|---|
| `evidence/evolution_proof.json` | `./evolution_proof.json` (giai đoạn 1, nguyên trạng) |
| `evidence/failure_mode_report.pdf` | `./runs/exp_self/failure_mode_report.pdf` |
| `evidence/learning_curve.pdf` | `./runs/exp_self/learning_curve.pdf` |
| `evidence/strategy_diversity.pdf` | `./runs/exp_self/strategy_diversity.pdf` |
| (bổ sung Phase 3) `runs/phase3/ablation_report.json`, `token_usage.json`, `run.log` | sinh tự động bởi pipeline |

## Phụ lục B — Checklist hoàn thiện gói nộp (tự kiểm trước khi ZIP)

- [ ] Điền toàn bộ ô 【Điền】 trong report này → convert sang `report.pdf`.
- [ ] `README.md` gói ZIP: team name, họ tên/MSSV/lớp/Kaggle username, submission cuối + LB score 0.77935, best dev 0.7346, commit hash, link video, khai báo dữ liệu.
- [ ] `integrity_declaration.pdf` — ký tên dưới tuyên bố liêm chính.
- [ ] `source_code/`: copy `src/`, `predict.py`, `requirements.txt` + viết `run_instructions.md` (lấy từ mục 6.2).
- [ ] `kaggle/`: `final_submission.csv` (bản 0.77935) + `submission_information.txt` (tên submission, thời điểm, score).
- [ ] Quay video 5-8 phút, up Drive, set "Anyone with the link can view", dán link vào README + report.
- [ ] Đặt tên ZIP: `A3_<KaggleTeamName>_<StudentID>.zip`.
