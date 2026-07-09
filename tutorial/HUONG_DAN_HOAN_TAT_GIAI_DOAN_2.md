# 🏁 Hướng dẫn hoàn tất Giai đoạn 2 (các bước còn lại)

> **Trạng thái tại thời điểm dừng (2026-07-09 ~04:30):**
> - ✅ Toàn bộ code pipeline giai đoạn 2 đã xong và đã test: `predict.py` + `src/{extraction_v2, pipeline_v2, voting, retrieval, judge}.py`, `complete_many` (retry/n>1), `evaluate_program_v2`. **6/6 graders vẫn PASS.**
> - ✅ **Dev 584 đã generate + repair XONG cả 2 strategies** (s4fix + retr, mỗi câu 16 candidates) — nằm an toàn trong `runs/phase3/dev_cache.jsonl`. Chạy lại lệnh generate sẽ tự resume, không tốn lại GPU.
> - ✅ Ablation đã chấm: **A1 = 0.6404** · **A3 = 0.7106** (baseline cũ: dev 0.579, LB 0.599).
> - ⬜ Còn lại: chấm A4 → test generation → submission. Chi tiết bên dưới.
>
> ⏰ **Deadline Kaggle: 2026-07-09 23:59 ICT (HÔM NAY).** Test generation cần ~3-6h GPU → nên bắt đầu Bước 2 **trước 14:00** (an toàn: trước trưa).

Mọi lệnh chạy từ thư mục `assignment03`, dùng Python của **Anaconda env `son_env`**:

```powershell
cd c:\Users\SonNT\Downloads\Advanced-NLP06\assignment03
$env:PYTHONUTF8 = "1"
$py = "C:\Users\SonNT\anaconda3\envs\son_env\python.exe"
```

---

## Bước 1 — Chấm A4 (ensemble) và quyết định cấu hình test (~1 phút, KHÔNG tốn GPU)

```powershell
& $py predict.py score --ablation A4
```

Đọc kết quả trong log (và `runs\phase3\ablation_report.json`):

| Tình huống | Quyết định cho test |
|---|---|
| **A4 > A3** (0.7106) | Test dùng **cả 2 strategies** → Bước 2 giữ nguyên `--strategies s4fix,retr` (~6h GPU) |
| **A4 ≤ A3** | Retrieval không giúp → test chỉ dùng **s4fix** → Bước 2 đổi thành `--strategies s4fix` và Bước 4/5 dùng `--ablation A3` (~3h GPU, tiết kiệm một nửa) |

> 💡 Nếu quỹ thời gian còn <7h mà A4 chỉ hơn A3 dưới ~1pp → chọn phương án s4fix-only cho chắc.

---

## Bước 2 — Test generation (~3-6h GPU) 🔥 BƯỚC TỐN THỜI GIAN NHẤT

```powershell
# Phương án A4 (2 strategies, ~6h):
& $py predict.py generate --split test --strategies s4fix,retr --groups greedy,sampled --k 7 --cache runs\phase3\test_cache.jsonl --concurrency 24

# HOẶC phương án A3 (chỉ s4fix, ~3h):
& $py predict.py generate --split test --strategies s4fix --groups greedy,sampled --k 7 --cache runs\phase3\test_cache.jsonl --concurrency 24
```

**Chú ý quan trọng:**
- **Bị đứt giữa chừng (mất mạng, tắt terminal, pod restart)? → chạy lại ĐÚNG lệnh đó.** Cache JSONL tự resume, chỉ chạy phần thiếu. Đây là thiết kế chủ đích — đừng xóa `test_cache.jsonl`.
- Đừng để máy tính sleep trong lúc chạy (loop chạy local, chỉ inference qua API).
- (Khuyến nghị) Trước run dài: vào Cerebrium Dashboard sửa `min_replicas = 1` để tránh cold start giữa chừng; **trả về 0 sau khi xong toàn bộ** để không tốn credits. Không sửa cũng được — retry tự xử lý.
- Theo dõi tiến độ: dòng tqdm `generate: x%` và `runs\phase3\run.log`.

Chạy xong thì repair (~10-15 phút):

```powershell
& $py predict.py repair --split test --cache runs\phase3\test_cache.jsonl --concurrency 12
```

---

## Bước 3 — Sinh submission AN TOÀN ngay lập tức (phao cứu sinh, ~1 phút)

Làm NGAY sau Bước 2, trước khi nghĩ đến judge:

```powershell
& $py predict.py submit --ablation A4 --cache runs\phase3\test_cache.jsonl --out runs\phase3\test_predictions.json
& $py format_submission.py --predictions runs\phase3\test_predictions.json --output-file submission.csv
& $py predict.py check --submission submission.csv
```

(Nếu Bước 1 chọn s4fix-only thì thay `--ablation A4` bằng `--ablation A3`.)

- `check` báo **CHECK PASS** → bạn đã có file submit hợp lệ trên đĩa. Từ giờ mọi thứ phía sau đều là "cộng thêm nếu kịp".
- Nếu check kêu có e-notation/-0.0 → chạy lại với `--fix`: `& $py predict.py check --submission submission.csv --fix` rồi check lại lần nữa.
- Nếu check FAIL vì số câu 0.0 > 10 → xem `runs\phase3\test_details.json` tìm các câu `no_candidates`, chạy lại repair rồi làm lại Bước 3.

---

## Bước 4 — (TÙY CHỌN — chỉ khi CÒN ≥4h trước deadline) Judge R1

**Điều kiện bật (đủ cả 3):** (a) đã có submission.csv PASS ở Bước 3; (b) còn ≥4h; (c) sẵn sàng chấp nhận rủi ro 2 lần deploy. **Thiếu 1 điều kiện → BỎ QUA, dùng file Bước 3.**

1. Thêm vào `.env`:
   ```
   JUDGE_BASE_URL=<giống CEREBRIUM_BASE_URL>
   JUDGE_API_KEY=<giống CEREBRIUM_API_KEY>
   JUDGE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
   JUDGE_MAX_CALLS=300
   ```
2. Sửa `cerebrium.toml` entrypoint: thay `QuantTrio/Qwen3.5-4B-AWQ` bằng `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` và `--gpu-memory-utilization 0.90` → `0.92` → `cerebrium deploy` (~10 phút + tải 15GB weights lần đầu).
3. Judge dev để xác nhận có lợi, rồi judge test:
   ```powershell
   & $py predict.py judge --split validation --cache runs\phase3\dev_cache.jsonl --ablation A4
   & $py predict.py score --ablation A5
   # CHỈ KHI A5 > A4 mới judge test:
   & $py predict.py judge --split test --cache runs\phase3\test_cache.jsonl --ablation A4
   ```
4. **Swap về Qwen**: hoàn nguyên `cerebrium.toml` (git checkout hoặc sửa tay lại như cũ) → `cerebrium deploy`.
5. Nếu A5 > A4: sinh lại submission với `--ablation A5` (lặp lại Bước 3 với ablation A5).

> Guard an toàn có sẵn: lệnh `judge` tự kiểm tra server đang serve đúng R1 (nếu quên swap sẽ báo lỗi ngay, không chấm bậy); tương tự `generate` từ chối chạy nếu server không phải Qwen.

---

## Bước 5 — Upload Kaggle + hậu kiểm (~15 phút)

1. Upload `submission.csv` lên Kaggle (**1 lần duy nhất** — file đã qua `check`).
2. Re-run 6 graders xác nhận điểm giai đoạn 1 còn nguyên:
   ```powershell
   & $py graders\grade_stage0.py; & $py graders\grade_stage1_executor.py; & $py graders\grade_stage2_reflector.py; & $py graders\grade_stage3_proposer.py; & $py graders\grade_smoke_proof.py; & $py graders\grade_stage4_harness.py
   ```
3. Nếu đã sửa `min_replicas` → trả về `0` + `cerebrium deploy`.
4. Bằng chứng cho report THINKFLIC nằm sẵn trong `runs\phase3\`: `run.log`, `phase3_config.json`, `ablation_report.json`, `ablation_*_eval.json`, `token_usage.json`, `test_details.json`, `test_predictions.json` + `submission.csv`. Nhớ khai báo trong report: pipeline (self-consistency K=8, ensemble retrieval few-shot, repair pass, judge nếu dùng), model (Qwen3.5-4B-AWQ + R1-7B nếu dùng), token/compute từ `token_usage.json`.

---

## 📊 Tham chiếu nhanh: số liệu & thời gian thực đo

| Việc | Thời gian thực đo | Kết quả |
|---|---|---|
| Dev s4fix (584 × greedy+K7) | 3h47 | runaway 19.5%, 0 request fail |
| Repair dev (987 mẫu) | 6.5 phút | vá được 660 |
| Dev retr (584 × greedy+K7) | ~3h40 | — |
| **A1** (greedy + fixes) | — | **0.6404** |
| **A3** (self-consistency) | — | **0.7106** |
| A4 / A5 | — | *chạy Bước 1 / Bước 4 để có số* |

**Kỳ vọng LB:** dev 0.711 (A3) ↔ LB dự kiến ~0.70-0.73; nếu A4/A5 cộng thêm thì tiến gần 0.75+. Baseline cũ 0.599.

## ⚠️ Các điểm cần chú ý chung

1. **Không xóa/sửa tay** các file trong `runs\phase3\*.jsonl` — đó là cache resume và bằng chứng.
2. **Không đụng** `runs\exp_self\`, `evolution_proof.json`, `graders\` — điểm giai đoạn 1 phụ thuộc chúng.
3. Mỗi lệnh `predict.py` tự ghi log vào `runs\phase3\run.log` — khi có lỗi, xem tail file này trước.
4. Lệnh nào cũng **chạy lại được an toàn** (idempotent qua cache) — khi nghi ngờ, cứ chạy lại.
5. Nếu Kaggle từ chối file: kiểm tra lại bằng `predict.py check` — file hợp lệ phải có đúng 3 cột `id,Usage,predicted_value` và 494 dòng đúng thứ tự `data\test.json`.
