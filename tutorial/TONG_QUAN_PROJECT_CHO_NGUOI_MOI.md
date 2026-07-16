# 📖 Tổng quan Project EvoAgent — dành cho người mới

> Tài liệu này giúp bạn **hiểu toàn bộ project trong ~15 phút đọc**: bài toán là gì, hệ thống
> hoạt động ra sao, file nào làm gì, và cần đọc tài liệu nào tiếp theo.
>
> Đây là tài liệu **giải thích khái niệm**. Còn hướng dẫn **thao tác từng bước** đã có sẵn:
> - Chạy & chấm điểm Giai đoạn 1 → [HUONG_DAN_CHOT_GIAI_DOAN_1.md](HUONG_DAN_CHOT_GIAI_DOAN_1.md)
> - Hoàn tất Giai đoạn 2 (Kaggle) → [HUONG_DAN_HOAN_TAT_GIAI_DOAN_2.md](HUONG_DAN_HOAN_TAT_GIAI_DOAN_2.md)

**Mục lục**

1. [Bài toán: dạy AI đọc báo cáo tài chính](#1-bài-toán-dạy-ai-đọc-báo-cáo-tài-chính)
2. [Vì sao bắt model viết "chương trình" thay vì trả lời thẳng?](#2-vì-sao-bắt-model-viết-chương-trình-thay-vì-trả-lời-thẳng)
3. [Ý tưởng cốt lõi: AI tự tiến hóa (EvoAgent)](#3-ý-tưởng-cốt-lõi-ai-tự-tiến-hóa-evoagent)
4. [Kiến trúc: cái gì chạy ở đâu?](#4-kiến-trúc-cái-gì-chạy-ở-đâu)
5. [Bản đồ thư mục & file](#5-bản-đồ-thư-mục--file)
6. [Hai giai đoạn của assignment](#6-hai-giai-đoạn-của-assignment)
7. [Vòng đời của một câu hỏi (luồng dữ liệu)](#7-vòng-đời-của-một-câu-hỏi-luồng-dữ-liệu)
8. [Từ điển thuật ngữ](#8-từ-điển-thuật-ngữ)
9. [Đọc gì tiếp theo?](#9-đọc-gì-tiếp-theo)

---

## 1. Bài toán: dạy AI đọc báo cáo tài chính

Đề bài (bối cảnh "công ty quant" trong [README](../README.md)): cho một đoạn trích **báo cáo
tài chính tiếng Việt** (văn bản + bảng số liệu) và một **câu hỏi tính toán**, hệ thống phải
tính ra **đáp án dạng số**.

Ví dụ (rút gọn):

> **Ngữ cảnh:** "Doanh thu năm 2016 là 100 tỷ đồng, năm 2017 là 108,5 tỷ đồng..."
> **Câu hỏi:** "Doanh thu 2017 tăng bao nhiêu phần trăm so với 2016?"

Model **không trả lời thẳng con số**, mà phải viết một **chương trình DSL** (ngôn ngữ mini
kiểu FinQA) mô tả cách tính:

```
subtract(108.5, 100), divide(#0, 100)
```

Máy sẽ thực thi chương trình này: bước 1 ra `8.5` (tham chiếu là `#0`), bước 2 lấy `#0 / 100`
ra `0.085` → đáp án **8,5%**. Đáp án được so với đáp án chuẩn (có dung sai làm tròn) để tính
accuracy.

Dữ liệu nằm trong `data/`: **train 2.986 câu, dev 584 câu, test 494 câu** (test bị giấu đáp
án — dùng cho Kaggle).

📚 *Chi tiết schema dữ liệu và cú pháp DSL đầy đủ:* [data_description.md](../data_description.md)
· [docs/DSL_SYNTAX.md](../docs/DSL_SYNTAX.md)

## 2. Vì sao bắt model viết "chương trình" thay vì trả lời thẳng?

Ba lý do, và hiểu chúng sẽ giúp bạn hiểu nhiều quyết định thiết kế trong code:

1. **LLM nhỏ tính nhẩm rất tệ.** Model dùng trong project chỉ 4B tham số — nhân chia số lẻ
   thường sai. Tách vai: model lo phần *lập luận* (chọn số nào, phép tính gì), máy tính lo
   phần *số học* (thực thi DSL trong [src/evaluator.py](../src/evaluator.py)).
2. **Chấm điểm được một cách khách quan.** Chương trình chạy ra một con số duy nhất — đúng
   hay sai rõ ràng, không cần người đọc lại lời giải.
3. **Kiểm tra được tính hợp lệ trước khi chạy.** Một chuỗi DSL có thể được validate bằng
   regex/parser (đây chính là việc của `_is_valid_dsl_program()` ở Stage 3 và của smoke test
   ở Stage 4).

## 3. Ý tưởng cốt lõi: AI tự tiến hóa (EvoAgent)

Prompt viết tay một lần thường không đủ tốt. Thay vì con người ngồi chỉnh prompt thủ công,
**EvoAgent để chính LLM chỉnh prompt cho LLM**, lặp đi lặp lại như một học sinh ôn thi:

| Học sinh ôn thi | EvoAgent | Code |
|---|---|---|
| Làm đề với "bí kíp" hiện tại | Chạy strategy trên train + dev, tính điểm | `executor.evaluate()` — Stage 1 |
| Xem lại bài sai, tìm quy luật lỗi | LLM tự phân tích lỗi, xuất JSON các *failure patterns* | `self_reflector.reflect_self()` — Stage 2 |
| Viết lại bí kíp dựa trên lỗi | LLM đề xuất prompt mới + few-shot nhắm đúng dạng bài sai | `self_proposer.propose_self()` — Stage 3 |
| Kiểm tra nhanh bí kíp mới có "đọc được" không | Smoke test: DSL hợp lệ? output không bị cụt? | `harness.run_smoke_test()` — Stage 4 |
| Chọn bí kíp tốt nhất từ trước đến giờ làm gốc | Chọn parent theo **AFO**: best / original / latest | `harness.select_parent_strategy()` — Stage 4 |

Vòng lặp (mỗi vòng = 1 *iteration*, chạy T vòng):

```
seed strategy (bí kíp khởi đầu — Stage 0)
        │
        ▼
┌─→ chọn parent ─→ đề xuất strategy mới ─→ smoke test ─→ đánh giá ─→ tự phản tư ─┐
│        (AFO)         (Stage 3)          (fail→thử lại)  (Stage 1)   (Stage 2)  │
└────────────────────────────────────────────────────────────────────────────────┘
     mỗi vòng ghi 1 dòng vào runs/.../history.jsonl → kỳ vọng dev accuracy tăng dần
```

Vài khái niệm quan trọng:

- **Strategy** (định nghĩa trong [src/strategy.py](../src/strategy.py)) = một "bí kíp" hoàn
  chỉnh: `prompt_template` (có chỗ trống `{passage}`, `{question}`, `{few_shot_block}`,
  `{cot_instruction}`), cách gợi suy luận (`cot_format`), giới hạn độ dài sinh
  (`max_new_tokens`)... kèm metadata: điểm dev/train, `parent_id` (gia phả), chi phí token.
- **Hai vai trò LLM:** *solver* (Qwen — giải bài) và *meta-agent* (phân tích lỗi + viết
  strategy mới). Trong project này cả hai vai đều do cùng model Qwen trên Cerebrium đảm nhận.
- **TokenBudget** ([src/executor.py](../src/executor.py)): sổ kế toán đếm token của cả hai
  vai, để phát hiện **runaway** (model lặp vô hạn, "suy nghĩ" mãi không dừng) — không phải
  hạn mức cứng cho cả run.

📚 *Sơ đồ mermaid chi tiết từng stage và mô tả nhiệm vụ từng file:* mục **Technical Briefing**
và **Phase 2** trong [README](../README.md).

## 4. Kiến trúc: cái gì chạy ở đâu?

Điểm hay gây bối rối nhất cho người mới: **vòng lặp EvoAgent chạy trên máy bạn**, chỉ có phần
suy luận của model là gọi qua mạng.

```
   MÁY CỦA BẠN (local, không cần GPU)              CEREBRIUM CLOUD (GPU A10 24GB)
┌───────────────────────────────────┐          ┌──────────────────────────────────┐
│ main.py / run_cerebrium.py /      │   HTTP   │  vLLM server (OpenAI-compatible) │
│ predict.py                        │ ────────▶│  model: QuantTrio/Qwen3.5-4B-AWQ │
│  └─ vòng lặp harness, chấm điểm,  │◀──────── │  scale-to-zero khi rảnh          │
│     thực thi DSL, ghi runs/, proof│  (src/model.py là client)                   │
└───────────────────────────────────┘          └──────────────────────────────────┘
```

Hệ quả thực tế:

- Cần file `.env` chứa `CEREBRIUM_BASE_URL`, `CEREBRIUM_API_KEY`, `CEREBRIUM_MODEL`,
  `HF_TOKEN` (đã gitignore — **không bao giờ commit**).
- Request đầu tiên sau khi server "ngủ" sẽ chậm 2–5 phút (**cold start**).
- Run dài phải giữ máy local không sleep; các run đều **resume** được từ cache/history.

📚 *Các bước deploy Cerebrium, điền `.env`, mẹo chi phí:* mục **Phase 1** trong
[README](../README.md) và Bước 1–4 trong [HUONG_DAN_CHOT_GIAI_DOAN_1.md](HUONG_DAN_CHOT_GIAI_DOAN_1.md).

## 5. Bản đồ thư mục & file

**Lối vào (entry points):**

| File | Vai trò |
|---|---|
| [main.py](../main.py) | CLI gốc chạy các chế độ của EvoAgent |
| [run_cerebrium.py](../run_cerebrium.py) | Script chạy các "job mốc" Giai đoạn 1: `sandbox` / `smoke` / `test` / `main` |
| [predict.py](../predict.py) | CLI pipeline Giai đoạn 2 (Kaggle): `generate` / `repair` / `score` / `submit` / `check` / `judge` |
| [submit.py](../submit.py), [format_submission.py](../format_submission.py) | Đóng gói predictions thành `submission.csv` đúng format Kaggle |

**Lõi EvoAgent — 5 file bạn phải implement ở Giai đoạn 1** (đều đã xong trong repo này):

| File | Stage | Nhiệm vụ |
|---|---|---|
| [src/sandbox.py](../src/sandbox.py) | 0 | Baseline zero-shot — đo "điểm xuất phát" chưa tối ưu |
| [src/executor.py](../src/executor.py) | 1 | Vòng đánh giá + `TokenBudget` |
| [src/self_reflector.py](../src/self_reflector.py) | 2 | Ép model xuất JSON phân loại lỗi của chính nó |
| [src/self_proposer.py](../src/self_proposer.py) | 3 | Sinh strategy mới + few-shot động từ các câu sai |
| [src/harness.py](../src/harness.py) | 4 | Vòng lặp tổng, smoke test, chọn parent (AFO) |

**Hạ tầng — không sửa:** [src/model.py](../src/model.py) (HTTP client gọi Cerebrium),
[src/evaluator.py](../src/evaluator.py) (thực thi DSL + so đáp án),
[src/strategy.py](../src/strategy.py) (dataclass), [src/data.py](../src/data.py) (load data),
[cerebrium.toml](../cerebrium.toml) (config deploy vLLM), `graders/` (bộ chấm tự động).

**Pipeline Giai đoạn 2 (Kaggle)** — xem mục 6:
[src/pipeline_v2.py](../src/pipeline_v2.py), [src/extraction_v2.py](../src/extraction_v2.py),
[src/voting.py](../src/voting.py), [src/retrieval.py](../src/retrieval.py),
[src/judge.py](../src/judge.py).

**Sản phẩm sinh ra khi chạy (không commit):**

- `runs/` — log, cache, lịch sử từng thí nghiệm (`exp_self/` = full run GĐ1, `phase3/` = GĐ2,
  mỗi run có `history.jsonl` + các file eval từng iteration).
- `sandbox_proof.json`, `smoke_proof.json`, `evolution_proof.json` — "bằng chứng" để grader
  xác nhận bạn đã chạy thật trên GPU.
- `submission.csv`, `submission_details.json` — file nộp Kaggle.
- `report/` — nháp báo cáo kỹ thuật cho gói nộp cuối.

## 6. Hai giai đoạn của assignment

### Giai đoạn 1 — Xây EvoAgent chuẩn (6 điểm)

Nhiệm vụ: điền các khối `TODO` trong 5 file lõi để "cỗ máy tự tiến hóa" ở mục 3 chạy được
thật. Mỗi bước dưới đây trả lời 3 câu: **nó làm gì — hoạt động thế nào — tại sao phải có nó**.

**Nhịp làm việc chung.** Với mỗi stage, bạn lặp 3 nhịp:

1. **Viết code** vào chỗ `TODO` (đặc tả và gợi ý đã ghi sẵn trong docstring).
2. **Chạy grader local** (`python graders/grade_stageX_*.py`) — grader giả lập model bằng
   mock nên chạy trong vài giây, **miễn phí**, bắt được lỗi logic ngay.
3. Khi code đúng hết, mới **chạy thật trên GPU** để sinh **file proof** (JSON bằng chứng).

*Tại sao tách làm 3 nhịp?* Vì GPU tính tiền theo giây. Nếu debug logic bằng cách chạy thật,
mỗi lần sửa một dòng code bạn đốt vài phút GPU. Grader mock cho phép sai thoải mái ở local;
GPU chỉ dùng khi mọi thứ đã chắc chắn. Còn file proof tồn tại vì grader không thể tự gọi
GPU của bạn — nó cần bằng chứng (kết quả, accuracy, số token, cấu hình run) rằng bạn đã
chạy thật chứ không bịa số.

**Stage 0 — Sandbox: đo điểm xuất phát.**
Cho model giải 50 câu dev ở chế độ *zero-shot* (hỏi trơn, không ví dụ mẫu, không tối ưu gì)
→ ra accuracy khoảng 40–42%. Hoạt động rất đơn giản: dựng prompt thô → gọi model → trích
chương trình DSL → thực thi → so đáp án.
*Tại sao cần?* Như kiểm tra đầu vào trước khóa học: không có mốc xuất phát thì về sau không
thể chứng minh vòng lặp tiến hóa **thực sự** cải thiện điều gì. Mọi con số của các stage sau
đều được so với mốc này.

**Stage 1 — Executor: máy chấm thi + sổ chi tiêu.**
Hai mảnh ghép:
- `evaluate()`: cho một strategy "đi thi" hàng loạt — chạy toàn bộ tập câu hỏi, chấm
  đúng/sai từng câu, tính accuracy tổng và **tách theo dạng bài** (cộng trừ đơn giản, đọc
  bảng, chia tỷ lệ...), đồng thời giữ lại danh sách câu sai kèm output gốc của model.
- `TokenBudget`: cuốn sổ ghi từng token vào/ra ở mọi lần gọi model, tách riêng token "giải
  bài" và token "meta" (phân tích lỗi, viết strategy). Nó cũng phát hiện **runaway** — khi
  model sinh lặp vô hạn không chịu dừng.
*Tại sao cần?* Vòng tiến hóa cần một thước đo khách quan để biết strategy mới hơn hay kém
hơn — cảm giác "prompt này trông xịn" không dùng được. Còn sổ token: mỗi token là tiền GPU;
một câu bị runaway có thể tốn bằng cả trăm câu bình thường, phải phát hiện được. Và danh
sách câu sai chính là **nguyên liệu đầu vào** cho Stage 2.

**Stage 2 — Reflector: buộc AI "tự thú" có cấu trúc.**
Sau mỗi lần thi, đưa các bài sai cho model và bắt nó tự phân tích: mình hay sai kiểu gì
(nhầm đơn vị, lấy sai số trong bảng, viết sai cú pháp DSL...), và giả thuyết vì sao.
Hoạt động qua **2 lượt hỏi**:
1. *Lượt nghĩ tự do* — model phân tích thoải mái bằng văn xuôi, được phép "suy nghĩ dài".
2. *Lượt ép khuôn* — yêu cầu đổ phân tích đó vào JSON đúng schema (danh sách failure
   patterns, giả thuyết, accuracy theo dạng bài). Server vLLM hỗ trợ *guided decoding*:
   ép model chỉ được sinh ra chuỗi khớp schema, nên JSON luôn parse được.
Nếu vẫn thất bại thì thử lại (tối đa 5 lần), mỗi lần **rút gọn dần ngữ cảnh** các bài sai
(cắt bớt đoạn văn dài) để prompt nhẹ hơn — đây là "progressive context decay".
*Tại sao cần?* Vì bước tiêu thụ kết quả này là **code**, không phải người. Code không đọc
được một đoạn văn tâm sự; nó cần các trường JSON cố định. Còn tách 2 lượt vì nếu ép JSON
ngay từ đầu, model dồn sức vào việc viết đúng cú pháp mà phân tích hời hợt — cho nghĩ tự do
trước, chất lượng phân tích tốt hơn hẳn.

**Stage 3 — Proposer: viết lại bí kíp dựa trên bằng chứng.**
Nhận bản JSON "tự thú" của Stage 2, đề xuất một strategy hoàn toàn mới: prompt mới, cách
gợi suy luận mới, và quan trọng nhất là **few-shot động** — nhìn xem model đang sai nhiều ở
dạng bài nào, lấy đúng vài câu dạng đó (kèm lời giải chuẩn) nhét vào prompt làm ví dụ mẫu.
Kèm theo hàm `_is_valid_dsl_program()` kiểm tra cú pháp mọi chương trình DSL trước khi dùng
làm ví dụ.
*Tại sao cần?* Đây chính là bước "tiến hóa" — không có nó thì hệ chỉ biết chấm điểm và than
thở chứ không tự sửa được. Few-shot phải *động* vì bốc ví dụ ngẫu nhiên là dạy lan man; sai
ở đọc bảng thì phải kèm ví dụ đọc bảng. Và phải validate DSL vì một ví dụ mẫu sai cú pháp
sẽ dạy model bắt chước cái sai — thuốc chữa biến thành thuốc độc.

**Stage 4 — Harness: nhạc trưởng + phanh an toàn.**
Ba mảnh ghép nối mọi thứ thành vòng lặp T vòng:
- `select_parent_strategy()` (AFO): đầu mỗi vòng, chọn strategy nào làm **nền để cải tiến
  tiếp** — cân nhắc giữa bản *tốt nhất từ trước tới nay*, bản *gốc*, và bản *mới nhất*.
  *Tại sao?* Tiến hóa không đi lên đều: strategy mới hoàn toàn có thể tệ hơn cũ. Nếu mù
  quáng lấy bản mới nhất làm nền, một vòng tồi sẽ kéo mọi vòng sau tụt theo. Giữ "file save
  tốt nhất" đảm bảo hệ không bao giờ mất thành quả đã đạt.
- `run_smoke_test()`: trước khi đem strategy mới đi chấm full (hàng trăm câu, tốn tiền),
  thử nhanh trên vài câu: có sinh ra DSL hợp lệ không? Output có bị cắt cụt giữa chừng
  không? Hỏng → bắt Stage 3 đề xuất lại (tối đa 3 lần).
  *Tại sao?* Như nếm thử một thìa canh trước khi múc cả nồi ra đãi khách — một prompt hỏng
  format mà lọt vào vòng chấm full là phí nguyên một lượt GPU chỉ để biết điều đã có thể
  biết sau 3 câu.
- `run_evoagent()`: vòng lặp tổng — mỗi vòng chạy đủ chọn-parent → đề-xuất → smoke →
  chấm-điểm → tự-thú, rồi ghi một dòng vào `history.jsonl` (nhật ký tiến hóa, để vẽ đường
  cong accuracy và để **resume** nếu đứt giữa chừng). Có chế độ **curriculum**: vòng đầu
  cho học câu dễ + đoạn văn ngắn, vòng 2 dồn câu khó (đọc bảng, nhiều bước tính), như học
  sinh học từ bài dễ đến bài khó.

**Vạch đích của giai đoạn.** Chạy 3 proof run trên GPU theo thứ tự: `sandbox` (Stage 0) →
`smoke` (kiểm tra hệ thống lắp ráp đúng) → `main` (run tiến hóa đầy đủ: 5 vòng, chấm trên
200 câu train + 240 câu dev, bật curriculum). Run cuối sinh `evolution_proof.json`; grader
yêu cầu **best dev accuracy ≥ 54%** — tức vòng lặp phải tự nâng model từ mốc zero-shot
~40–42% lên thêm ít nhất ~12 điểm phần trăm, **không có con người nào chỉnh prompt bằng
tay**. Đó chính là điều Giai đoạn 1 muốn chứng minh: bạn đã xây được một hệ tự cải thiện
đo đếm được, làm bệ phóng cho giai đoạn tự do sáng tạo phía sau.

### Giai đoạn 2 — Kaggle leaderboard (4 điểm)

**Cuộc chơi thay đổi thế nào.** Giai đoạn 1 chấm việc bạn xây máy đúng đặc tả; Giai đoạn 2
chỉ chấm đúng một thứ: **accuracy trên 494 câu test bị giấu đáp án**. Bạn nộp file CSV dự
đoán (mỗi dòng: id câu hỏi + con số) lên Kaggle; điểm cuối tính trên *private leaderboard*
— phần test được giữ kín, chỉ công bố sau deadline, để không ai "học tủ" theo bảng điểm
public. Luật mở hoàn toàn: mọi kỹ thuật tăng accuracy đều hợp lệ. Repo này tự đặt thêm một
ràng buộc: **không sửa code Giai đoạn 1** (kẻo hỏng điểm grader) — toàn bộ pipeline mới nằm
ở các file `extraction_v2.py`, `voting.py`, `retrieval.py`, `pipeline_v2.py`, `judge.py`
và CLI `predict.py`.

**Điểm xuất phát và hai đòn bẩy.** Lấy strategy tốt nhất mà run tiến hóa Giai đoạn 1 tìm ra
(tên nội bộ `s4fix`) chạy thẳng trên dev → **0.579**. Soi kỹ các câu sai thì thấy chúng chia
làm hai loại rất khác nhau:
- *(a)* model **giải đúng nhưng hệ thống đọc sai** — chương trình DSL nằm đâu đó trong
  output mà bước trích xuất không lấy ra được, hoặc lấy nhầm rác;
- *(b)* model **giải sai thật**.

Hai loại lỗi → hai đòn bẩy độc lập: *"đừng đánh rơi câu model đã làm đúng"* và *"tăng xác
suất model làm đúng"*. Toàn bộ Giai đoạn 2 là lắp dần từng đòn bẩy và **đo** xem mỗi cái
đáng bao nhiêu điểm.

**Đòn bẩy 1 — Extraction v2 + Repair: đừng đánh rơi câu đúng.**
Bộ trích cũ rất "tham": gom mọi mảnh `op(...)` tìm thấy ở bất kỳ đâu trong output rồi nối
lại — khi model bị runaway (nghĩ lan man không dừng), nó nối rác thành chương trình rác.
Bản v2 thay bằng **thang ưu tiên** nghiêm ngặt: output chưa đóng thẻ suy nghĩ `</think>` →
tuyên bố runaway và *không vớt vát gì cả*; còn lại thì tìm theo thứ tự: khối JSON sạch nằm
sau `</think>` (đường "khỏe mạnh", phủ gần hết output tử tế) → JSON bị cắt cụt vì chạm trần
token → JSON nằm chỗ khác → dòng cuối cùng chỉ chứa thuần chương trình. Nấc nào cũng phải
qua kiểm tra cú pháp mới được nhận.
Những câu vẫn trắng tay sau đó đi qua **repair pass**: gửi lại chính output hỏng cho model
kèm prompt "đây là bài làm dở dang của bạn, hãy chốt lại chương trình" và ép trả lời đúng
khuôn JSON. Trên dev, repair vá được 660 trong số 987 mẫu hỏng, hết ~7 phút.
*Tại sao đáng làm trước tiên?* Vì sửa cách **đọc** output không tốn GPU sinh mới — chấm lại
từ output có sẵn. Riêng đòn bẩy này đưa dev từ 0.579 lên **0.640**: gần 6 điểm phần trăm
vốn dĩ là của mình, chỉ bị đánh rơi ở khâu trích xuất.

**Đòn bẩy 2 — Self-consistency: hỏi 8 lần, lấy đáp án đa số.**
Thay vì hỏi mỗi câu một lần, sinh **8 lời giải**: 1 bản *greedy* (model trả lời "chắc tay"
nhất) + 7 bản *sampled* (bật temperature cho mỗi lần suy luận đi một ngả). Thực thi cả 8
chương trình → gom các kết quả xấp xỉ nhau thành **cụm** (hai số coi là một nếu chênh dưới
0,5% hoặc 1e-4) → **bầu chọn có trọng số**: phiếu greedy nặng hơn một chút, lời giải nào mà
con số model tự tuyên bố khớp với kết quả thực thi cũng được cộng điểm tin cậy → lấy cụm
nặng nhất làm đáp án.
*Tại sao hiệu quả?* Một lần suy luận có thể trượt ngẫu nhiên (lấy nhầm hàng trong bảng, lỡ
một bước tính), nhưng các lần trượt thường trượt *khác nhau*, còn các lần đúng thì *cùng ra
một số*. Nếu 6/8 lời giải độc lập hội tụ về một giá trị, giá trị đó gần như chắc đúng —
"hỏi ý kiến đám đông" mà đám đông là chính mình. Đòn bẩy này đưa 0.640 lên **0.711**. Giá
phải trả: tốn ~8 lần GPU so với hỏi một lần — vì vậy mới cần đo trước khi tiêu (xem ablation
bên dưới).

**Đòn bẩy 2.5 — Retrieval few-shot (đội hình ensemble).**
Thêm một "chiến lược thứ hai" tên `retr`: dùng cùng prompt nền như `s4fix` nhưng ví dụ mẫu
được chọn **riêng cho từng câu hỏi** — tìm những câu train giống câu đang hỏi nhất (so khớp
TF-IDF trên n-gram ký tự, cách so chịu được dấu tiếng Việt mà không cần thư viện nặng), rồi
lấy chương trình gold của chúng làm bài mẫu. 8 lời giải của `retr` đổ chung "nồi phiếu" với
8 lời giải của `s4fix` — mỗi câu 16 phiếu.
*Tại sao?* Bộ ví dụ cố định không thể phủ hết mọi dạng bài; cho model xem một bài *gần y
hệt* đã giải sẵn là cách dạy format chính xác nhất mà không tốn công huấn luyện. Và hai
nguồn lời giải độc lập thường sai theo *kiểu khác nhau*, nên bầu chung càng khó bị một kiểu
sai kéo lệch.

**Trọng tài phút chót — LLM-as-judge (tùy chọn).**
Cuộc bầu chọn tự biết khi nào mình **không chắc**: cụm thắng chiếm dưới nửa trọng số phiếu,
có ít hơn 3 phiếu, hoặc chỉ nhỉnh hơn cụm nhì sát nút. Chỉ những câu "nghi ngờ" đó (và giới
hạn tối đa 300 câu) mới được gửi cho một model suy luận mạnh hơn — DeepSeek-R1-7B, tạm swap
vào chạy trên chính app Cerebrium sau khi Qwen sinh xong — để phân xử: chọn cụm nào, hoặc
tự viết chương trình sửa lại. Hai chi tiết thiết kế đáng học:
- R1 được **nghĩ tự do** trước, phán quyết ghi *sau* thẻ `</think>` theo 3 dòng chữ đơn giản
  (`SELECTION` / `PROGRAM` / `FINAL_VALUE`) thay vì ép JSON ngay — vì ép model suy luận xuất
  JSON tức thì làm chất lượng lập luận của nó rơi 30–40%.
- Có **khóa an toàn**: lệnh judge tự kiểm tra server đang chạy đúng R1 (và lệnh generate từ
  chối chạy nếu server không phải Qwen) — quên swap model là bị chặn ngay, không chấm bậy.
*Tại sao chỉ chấm câu nghi ngờ?* Judge đắt (model to hơn, phải suy luận dài). Với câu mà 16
phiếu đã đồng thuận thì judge gần như không đổi được gì — tiền chỉ nên tiêu ở đúng chỗ hệ
thống đang phân vân.

**Ablation A1→A5: bật dần từng đèn để biết đèn nào sáng.**
Mỗi cấu hình chồng thêm một kỹ thuật: **A1** = chỉ `s4fix` greedy + extraction v2 (0.640) →
**A3** = thêm self-consistency 8 phiếu (0.711) → **A4** = thêm `retr` vào ensemble (16
phiếu) → **A5** = thêm judge. Điều then chốt: sinh lời giải cho *test* tốn 3–6 giờ GPU,
nhưng **chấm lại dev từ cache thì miễn phí** — nên mọi quyết định "test dùng cấu hình nào"
đều được đo trên dev trước bằng con số, không bằng cảm tính. Kỹ thuật nào không cộng thêm
điểm thì loại khỏi run test, tiết kiệm được một nửa GPU.

**Xương sống kỹ thuật — cache & resume.**
Mỗi lời giải sinh ra được ghi *ngay lập tức* thành một dòng JSONL với khóa (câu hỏi, chiến
lược, mẫu thứ mấy). Chạy lại đúng lệnh cũ → hệ thống đọc cache, chỉ chạy phần còn thiếu.
*Tại sao đây là mảnh quan trọng nhất dù "không thông minh"?* Vì run nhiều giờ qua mạng
*chắc chắn* có lúc đứt (rớt wifi, máy sleep, server restart). Không có cache, một sự cố ở
giờ thứ 5 nghĩa là mất trắng 5 giờ GPU; có cache, nó chỉ là phiền toái vài phút. Nguyên tắc:
mọi lệnh đều chạy lại được an toàn — khi nghi ngờ, cứ chạy lại.

**Đường ra trận (trình tự thực tế).** Sinh lời giải cho dev → repair → chấm A1..A5 → chốt
cấu hình → sinh lời giải cho test (bước tốn thời gian nhất) → repair → bầu chọn ra đáp án
từng câu → xuất CSV đúng 494 dòng theo thứ tự file test → chạy lệnh `check` bắt các lỗi
format hay gây tạch oan (số dạng `1e-05`, `-0.0`, thiếu dòng) → nộp Kaggle **một lần duy
nhất** bằng file đã qua check. Kết quả của repo này: dev 0.579 → **0.711**, so với baseline
leaderboard 0.599.

## 7. Vòng đời của một câu hỏi (luồng dữ liệu)

Ghép tất cả lại, một câu hỏi trong `dev.json` đi qua hệ thống như sau:

1. **Load** ([src/data.py](../src/data.py)): đọc item JSON — gồm `pre_text`, bảng, `post_text`,
   câu hỏi, đáp án gold.
2. **Dựng prompt**: đổ ngữ cảnh + câu hỏi + few-shot vào `prompt_template` của strategy
   hiện hành.
3. **Gọi model** ([src/model.py](../src/model.py)): gửi prompt qua HTTP đến vLLM/Qwen trên
   Cerebrium, nhận về văn bản (thường gồm phần suy luận + chương trình DSL).
4. **Trích chương trình**: lọc chuỗi DSL ra khỏi văn bản (GĐ1: `extract_answer`;
   GĐ2: extraction v2 + repair).
5. **Thực thi & chấm** ([src/evaluator.py](../src/evaluator.py)): chạy DSL từng bước
   → ra một con số → so với gold (có dung sai) → đúng/sai.
6. **Ghi sổ**: cộng token vào `TokenBudget`, ghi kết quả vào `runs/`; ở GĐ1 các câu sai được
   đưa cho reflector phân tích để đề xuất strategy tốt hơn ở vòng sau; ở GĐ2 nhiều candidate
   của cùng câu được đưa vào voting.

## 8. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa trong project này |
|---|---|
| **DSL** | Ngôn ngữ mini kiểu FinQA: chuỗi phép tính phẳng `op(a,b), op(#0,c)...` — xem [docs/DSL_SYNTAX.md](../docs/DSL_SYNTAX.md) |
| **Seed strategy** | Strategy khởi đầu viết sẵn, điểm xuất phát của tiến hóa |
| **Iteration (T)** | Một vòng lặp tiến hóa hoàn chỉnh (propose → smoke → evaluate → reflect) |
| **AFO** | Cách chọn parent cho vòng kế: cân nhắc giữa strategy *best* / *original* / *latest* để tránh thoái hóa |
| **Smoke test** | Kiểm tra nhanh trên vài mẫu: strategy mới có sinh DSL hợp lệ, không bị cắt cụt? Fail thì đề xuất lại (tối đa 3 lần) |
| **CoT (chain-of-thought)** | Cho model "nghĩ thành lời" trước khi chốt chương trình; đổi bằng token — đó là lý do có nấc `max_new_tokens` theo strategy |
| **Runaway** | Model sinh lặp vô hạn không chịu dừng — bị phát hiện qua TokenBudget và bị extraction v2 từ chối vớt vát |
| **Zero-shot / Few-shot** | Hỏi không kèm ví dụ mẫu / kèm vài ví dụ mẫu trong prompt |
| **Self-consistency** | Sinh nhiều lời giải rồi lấy đáp án đa số (bầu chọn) |
| **Ablation (A1, A3, A4, A5)** | Các cấu hình bật dần từng kỹ thuật để đo đóng góp riêng của nó |
| **Proof file** | JSON bằng chứng run thật trên GPU (`sandbox_proof` / `smoke_proof` / `evolution_proof`) mà grader kiểm tra |
| **Grader** | Script trong `graders/` tự chấm code + proof của bạn, chạy local |
| **Cold start** | Server GPU scale về 0 khi rảnh; request đầu tiên phải chờ nó khởi động lại (2–5 phút) |

## 9. Đọc gì tiếp theo?

| Bạn muốn... | Đọc |
|---|---|
| Hiểu đề bài gốc, checklist điểm số | [README.md](../README.md) |
| Hiểu format dữ liệu & DSL sâu | [data_description.md](../data_description.md) · [docs/DSL_SYNTAX.md](../docs/DSL_SYNTAX.md) · [docs/DATA_DESCRIPTION.md](../docs/DATA_DESCRIPTION.md) |
| Tự tay chạy lại Giai đoạn 1 | [HUONG_DAN_CHOT_GIAI_DOAN_1.md](HUONG_DAN_CHOT_GIAI_DOAN_1.md) |
| Chạy pipeline Kaggle / tái tạo submission | [HUONG_DAN_HOAN_TAT_GIAI_DOAN_2.md](HUONG_DAN_HOAN_TAT_GIAI_DOAN_2.md) |
| Luật Kaggle, format nộp, gói nộp cuối | [docs/PHASE3_KAGGLE.md](../docs/PHASE3_KAGGLE.md) · [docs/SUBMISSION_FORMAT.md](../docs/SUBMISSION_FORMAT.md) · [docs/THINKFLIC_SUBMISSION.md](../docs/THINKFLIC_SUBMISSION.md) |
| Đọc code theo thứ tự dễ hiểu nhất | `strategy.py` → `model.py` → `evaluator.py` → `sandbox.py` → `executor.py` → `self_reflector.py` → `self_proposer.py` → `harness.py` → (GĐ2) `extraction_v2.py` → `voting.py` → `retrieval.py` → `pipeline_v2.py` |
| Xem kết quả & phân tích đã chạy | `runs/exp_self/history.jsonl` (đường cong tiến hóa GĐ1) · `runs/phase3/ablation_report.json` (điểm từng ablation GĐ2) · [report/technical_report_draft.md](../report/technical_report_draft.md) |
