# 🎯 Hướng dẫn từng bước chốt Giai đoạn 1 (Proof Runs trên Cerebrium)

> **Trạng thái hiện tại (đã xong):**
> - ✅ Toàn bộ code Stage 0–4 đã implement (`src/sandbox.py`, `executor.py`, `self_reflector.py`, `self_proposer.py`, `harness.py`)
> - ✅ Tất cả grader **logic** đã PASS (chạy local với mock, không cần GPU)
> - ✅ `src/model.py` đã là HTTP client OpenAI-compatible (gọi Cerebrium thay vì load model local)
> - ✅ Venv `son_env` + dependencies đã cài, `HF_TOKEN` trong `.env` đã xác thực
>
> **Còn thiếu để chốt điểm:** 3 file proof do chạy thật trên GPU sinh ra:
> `sandbox_proof.json`, `smoke_proof.json`, `evolution_proof.json` (cần best dev accuracy ≥ 54%).

Tổng thời gian dự kiến: **~30 phút thao tác** + **vài giờ chờ** full run chạy.

---

## Bước 0 — Mở terminal đúng môi trường

Mọi lệnh trong tài liệu này chạy bằng **PowerShell**, trong thư mục `assignment03`, với venv `son_env`:

```powershell
cd c:\Users\SonNT\Downloads\Advanced-NLP06\assignment03
.\son_env\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"    # để grader in tiếng Việt không lỗi Unicode
```

> 💡 Set `$env:PYTHONUTF8="1"` lại mỗi khi mở terminal mới (hoặc set vĩnh viễn:
> `[Environment]::SetEnvironmentVariable("PYTHONUTF8","1","User")`).

---

## Bước 1 — Tạo tài khoản & cài Cerebrium CLI (~5 phút)

1. Vào <https://cerebrium.ai/> → **Sign up** (đăng nhập bằng Google/GitHub đều được).
2. Sau khi vào Dashboard, Cerebrium sẽ tạo cho bạn một **Project** (id dạng `p-xxxxxxxx`) — ghi nhớ id này, lát nữa cần cho `CEREBRIUM_BASE_URL`.
3. Cài CLI và đăng nhập (trong venv `son_env`):

```powershell
pip install cerebrium
cerebrium login
```

Lệnh `cerebrium login` sẽ mở trình duyệt để xác thực. Kiểm tra thành công:

```powershell
cerebrium project list
```

> 💰 **Lưu ý chi phí:** Cerebrium tính tiền theo giây GPU chạy. App được cấu hình
> `min_replicas = 0` (scale về 0 khi rảnh → không tốn tiền khi không dùng).
> Theo dõi usage tại Dashboard → **Billing**.

---

## Bước 2 — Thêm secret `HF_TOKEN` trên Cerebrium Dashboard (~2 phút)

Server vLLM cần token HuggingFace để tải model weights (`QuantTrio/Qwen3.5-4B-AWQ`).

1. Vào **Cerebrium Dashboard** → chọn project của bạn → menu **Secrets**.
2. Bấm **Add Secret**:
   - **Name:** `HF_TOKEN`
   - **Value:** token trong file `.env` của bạn (dòng `HF_TOKEN=hf_...`)
3. Save.

> ⚠️ Không hardcode token vào `cerebrium.toml` hay commit lên git.

---

## Bước 3 — Deploy vLLM server (~10–15 phút lần đầu)

Từ thư mục `assignment03` (nơi có sẵn `cerebrium.toml`):

```powershell
cerebrium deploy
```

- Lần đầu deploy sẽ lâu (kéo image vLLM + tải ~3GB model weights). Weights được
  cache vào persistent storage nên các lần cold start sau nhanh hơn nhiều.
- Deploy xong, CLI in ra **endpoint URL** của app, dạng:

```
https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxxxx/assignment03
```

**Lấy API key:** Dashboard → project → **API Keys** → copy **Inference Token** (JWT).

> 🔧 Nếu deploy lỗi vì vLLM chưa hỗ trợ Qwen3.5: mở `cerebrium.toml`, đổi model
> trong `entrypoint` thành `Qwen/Qwen3-4B-AWQ`, deploy lại, và nhớ đổi
> `CEREBRIUM_MODEL` trong `.env` tương ứng (Bước 4).

---

## Bước 4 — Điền `.env` (~1 phút)

Mở file `assignment03\.env` và điền 2 dòng còn trống (**thêm `/v1` vào cuối base URL**):

```
HF_TOKEN=hf_...                          # đã có sẵn
CEREBRIUM_BASE_URL=https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxxxx/assignment03/v1
CEREBRIUM_API_KEY=<Inference Token vừa copy>
CEREBRIUM_MODEL=QuantTrio/Qwen3.5-4B-AWQ
```

**Kiểm tra endpoint sống** (lệnh này cũng "đánh thức" replica nếu đang scale về 0 — lần đầu có thể chờ 2–5 phút cold start):

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; from openai import OpenAI; c = OpenAI(base_url=os.environ['CEREBRIUM_BASE_URL'], api_key=os.environ['CEREBRIUM_API_KEY'], timeout=600); print([m.id for m in c.models.list().data])"
```

Kết quả mong đợi: `['QuantTrio/Qwen3.5-4B-AWQ']`. Nếu ra đúng → hạ tầng xong 100%.

---

## Bước 5 — Stage 0: Sandbox proof (~5–10 phút GPU)

Chạy zero-shot baseline trên 50 ví dụ dev, sinh `sandbox_proof.json`:

```powershell
python run_cerebrium.py sandbox
```

Rồi chấm điểm:

```powershell
python graders\grade_stage0.py
```

Kỳ vọng: **SUCCESS! STAGE 0 COMPLETELY VERIFIED** (cả test logic lẫn test proof đều pass).

---

## Bước 6 — Stage 4: Smoke proof (~5 phút GPU)

Chạy pre-flight smoke test trên seed strategy, sinh `smoke_proof.json`:

```powershell
python run_cerebrium.py smoke
python graders\grade_smoke_proof.py
```

Kỳ vọng: `smoke_test_passed: true`, grader báo PASS.

---

## Bước 7 — Confidence run nhỏ (khuyến nghị, ~20–40 phút GPU)

Trước khi đốt credits cho full run, chạy bản nhỏ (train/dev = 32) để bắt lỗi tích hợp:

```powershell
python run_cerebrium.py test
```

Xem kết quả trong `runs\exp_test\` (file `history.jsonl` và các `iter_XXX_eval_dev.json`).
Nếu loop chạy đủ 5 iteration không crash và dev accuracy trông hợp lý → sẵn sàng full run.

> 💡 **Mẹo tránh cold start giữa chừng:** trước run dài, mở `cerebrium.toml`
> sửa `min_replicas = 1` rồi `cerebrium deploy` lại. **Nhớ trả về `0` sau khi
> chạy xong** để không tốn tiền khi rảnh.

---

## Bước 8 — Full proof run (~vài giờ GPU) 🏁

Đây là run sinh `evolution_proof.json` (T=5, train 200, dev 240, curriculum + AFO best):

```powershell
python run_cerebrium.py main
```

- Run chạy **local trên máy bạn** (chỉ inference qua API) → đừng tắt máy/sleep giữa chừng.
- Nếu bị gián đoạn: chạy lại với `python run_cerebrium.py main --resume` — loop tự
  resume từ `runs\exp_self\history.jsonl`, không mất iteration đã xong.
- Kết thúc, script in `Best dev accuracy: ...` và ghi `evolution_proof.json` ngay
  tại thư mục `assignment03`.

Chấm điểm cuối:

```powershell
python graders\grade_stage4_harness.py
```

Kỳ vọng: PASS toàn bộ 4 test, trong đó `test_evolution_proof` yêu cầu
**best dev accuracy ≥ 54%** (baseline 42%).

> ❗ **Nếu accuracy < 54%:** chưa đạt thì đừng hoảng — vài hướng chỉnh nhanh:
> - Chạy lại `main` (temperature 0.7 ở bước propose nên mỗi run tiến hóa khác nhau).
> - Tăng `--T` (số iteration) trong `run_cerebrium.py` (hàm `run_main`) từ 5 → 7.
> - Xem `runs\exp_self\iter_XXX_reflection.json` để hiểu model đang yếu ở đâu.

---

## Bước 9 — Tổng kiểm tra & chốt sổ ✅

Chạy lại đủ 6 grader — tất cả phải PASS:

```powershell
python graders\grade_stage0.py
python graders\grade_stage1_executor.py
python graders\grade_stage2_reflector.py
python graders\grade_stage3_proposer.py
python graders\grade_smoke_proof.py
python graders\grade_stage4_harness.py
```

Sau đó:

1. **Trả `min_replicas` về `0`** trong `cerebrium.toml` (nếu đã tăng ở Bước 7) và `cerebrium deploy` lại — tránh tốn credits.
2. Đọc `docs\THINKFLIC_SUBMISSION.md` để đóng gói hồ sơ nộp (report, evidence, video).
3. 🎉 Giai đoạn 1 hoàn tất — Giai đoạn 2 (Phase 3 Kaggle) xem `docs\PHASE3_KAGGLE.md`.

---

## 🔧 Troubleshooting nhanh

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `No inference endpoint configured` | `.env` thiếu `CEREBRIUM_BASE_URL` hoặc chạy lệnh ngoài thư mục `assignment03`. |
| Request đầu tiên treo 2–5 phút | Cold start (replica đang scale từ 0). Client đã set timeout 600s — cứ chờ, hoặc set `min_replicas = 1`. |
| `401 Unauthorized` | Sai `CEREBRIUM_API_KEY` — lấy lại Inference Token từ Dashboard → API Keys. |
| `404` khi gọi models.list | Base URL sai — kiểm tra đúng dạng `.../v4/p-<PROJECT_ID>/assignment03/v1` (đủ hậu tố `/v1`). |
| Deploy lỗi tải model / vLLM không nhận Qwen3.5 | Đổi model sang `Qwen/Qwen3-4B-AWQ` ở cả `cerebrium.toml` lẫn `CEREBRIUM_MODEL` trong `.env`. |
| Grader in lỗi Unicode trên Windows | Quên `$env:PYTHONUTF8="1"`. |
| Full run bị đứt giữa chừng | `python run_cerebrium.py main --resume` để chạy tiếp từ history. |
