# Technical Report — Assignment 03: Self-Improving AI for Financial Question Answering

> **Cách đọc report này:** các hộp `💡` giải thích thuật ngữ cho người đọc mới tiếp cận LLM/Agent; mỗi khối kỹ thuật đều ghi rõ nó nằm ở **file nào** trong source code (đường dẫn tính từ thư mục gốc `assignment03/`), kèm ví dụ cụ thể lấy từ dữ liệu và code thật. Phụ lục B là "bản đồ codebase" để tra nhanh.

---

## 1. Thông tin cá nhân

| Mục | Thông tin |
|---|---|
| Họ tên | Nguyễn Trường Sơn |
| Kaggle username | truongson1209 |
| Kaggle team name | Son Nguyen |
| Hình thức nộp | Cá nhân (100% khối lượng công việc) |
| Video thuyết trình | https://drive.google.com/file/d/1Wkg4p0VsM9F1yfeT5RGDlOCynPAdNrqS/view?usp=sharing |

---

## 2. Tóm tắt & Kết quả Kaggle

**Bài toán** (FinQA-style, tiếng Việt): mỗi câu hỏi đi kèm một trang trích từ báo cáo tài chính — gồm đoạn văn và bảng số liệu — ví dụ *"Tỷ lệ tăng trưởng giá cổ phiếu của PMI từ năm 2012 đến 2013 là bao nhiêu?"*. Hệ thống phải trả về **đúng một con số**. Thay vì để model trả lời thẳng (khó kiểm chứng, dễ tính sai), hệ thống EvoAgent yêu cầu model sinh một **chương trình DSL phẳng** mô tả cách tính (`subtract(a,b), divide(#0,b)`...), rồi tự thực thi chương trình đó để ra giá trị cuối cùng — sai ở bước nào là nhìn thấy được ở bước đó.

> 💡 **DSL (Domain-Specific Language) là gì?** Một "ngôn ngữ lập trình mini" chỉ có ~11 phép tính (`add, subtract, multiply, divide, exp, greater, abs` + 4 phép trên bảng `table_max/min/sum/average`). "Phẳng" nghĩa là mỗi bước một phép tính, không lồng nhau; bước sau tham chiếu kết quả bước trước bằng `#0`, `#1`... Ví dụ tính tăng trưởng từ 100 lên 108.50: `subtract(108.50, 100), divide(#0, 100)` → bước 0 ra 8.5, bước 1 lấy `#0` chia 100 ra `0.085`. Cú pháp đầy đủ ở `docs/DSL_SYNTAX.md`, phần thực thi ở `src/evaluator.py`.

**Kết quả chính:**

| Mốc | Dev accuracy | Kaggle Public LB |
|---|---|---|
| Baseline (EvoAgent thuần, iter_004, greedy) | 0.5792 (dev-240) | **0.59919** |
| Hệ thống cuối (Giai đoạn 3) | **0.7346** (full dev-584) | **0.77935** |
| Cải thiện | +15.5 pp | **+18.0 pp** |

Submission cuối: 【Điền: tên submission trên Kaggle】 · Rank tại thời điểm nộp: 【Điền: rank/tổng số team】 (top LB lúc đó: 0.80971).

**Ba nguồn cải thiện chính** (các kỹ thuật in nghiêng sẽ được giải thích lần lượt ở mục 3):

1. Vá các lỗi hệ thống của baseline — request thất bại bị nuốt thành 0.0, chuỗi suy nghĩ chạy vô hạn, extraction gom rác, few-shot dạy sai cú pháp (~+6 pp).
2. *Self-consistency voting* trên 16 candidates/câu (~+7 pp).
3. Ensemble với *retrieval few-shot* per-question (~+2.4 pp).

---

## 3. Pipeline cuối cùng

Quy ước tên gọi theo đề bài: **Giai đoạn 2** = tái dựng vòng lặp tự cải thiện EvoAgent theo các khối TODO (Milestone 1 — chạy tiến hóa strategy, sinh `evolution_proof.json`, strategy tốt nhất `iter_004`); **Giai đoạn 3** = cuộc thi Kaggle mở, tự do kỹ thuật.

Hệ thống cuối của Giai đoạn 3 không xây từ con số 0: nó đứng trên bốn "tài sản" của Giai đoạn 2 (mục 3.2), rồi bọc quanh đó một chuỗi khối xử lý mới (mục 3.3).

### 3.1 Nhìn toàn cảnh: một câu hỏi đi từ đầu đến cuối

**Ví dụ xuyên suốt** (mẫu thật trong `data/train.json`, id `PM/2017/page_25.pdf-1`): ngữ cảnh gồm đoạn văn + bảng giá cổ phiếu PMI, trong đó có hai ô «31/12/2012: $100.00» và «31/12/2013: $108.50». Câu hỏi: *"Tỷ lệ tăng trưởng giá cổ phiếu của PMI từ năm 2012 đến 2013 là bao nhiêu?"*.

Model không được trả lời thẳng "8.5%" mà phải sinh **chương trình DSL phẳng**:

```
subtract(108.50, 100), divide(#0, 100)
```

(`#0` = kết quả bước trước). Hệ thống thực thi chương trình để ra giá trị nộp bài `0.085` — khớp đúng đáp án gold (`exe_ans: 0.085`) của mẫu này. Mọi khối dưới đây đều phục vụ một mục tiêu duy nhất: **với mỗi câu hỏi, thu về một chương trình chạy được và đúng.**

Dòng chảy đầy đủ cho mỗi câu trong 494 câu test:

```
test.json (câu hỏi + văn bản + bảng)
   ▼
[1] Dựng prompt ×2 strategy (s4fix / retr)     ← tài sản Giai đoạn 2: template + few-shot iter_004
   ▼
[2] Sinh 16 candidates  = 2 strategy × (1 greedy + 7 sampled)   — vLLM trên Cerebrium
   ▼
[3] Extraction v2: bóc chương trình DSL từ raw output
   ▼
[4] Repair: sample nào bóc thất bại → re-ask guided-JSON 1 lần
   ▼
[5] Evaluator v2: THỰC THI từng chương trình → giá trị số
   ▼
[6] Voting self-consistency: cluster theo giá trị → chọn cluster thắng
   ▼
[7] format_submission.py → predict.py check → submission.csv → Kaggle
```

> 💡 **Prompt, few-shot là gì?** *Prompt* là toàn bộ văn bản gửi cho model mỗi lần hỏi: lời dặn hệ thống ("bạn là chuyên gia phân tích tài chính..."), vài **bài mẫu đã giải sẵn** (gọi là *few-shot* — model nhìn bài mẫu để bắt chước format và cách giải), rồi mới đến ngữ cảnh + câu hỏi thật. Model không được "huấn luyện" gì thêm — nó chỉ đọc prompt và viết tiếp.

Mỗi bước trong sơ đồ ứng với đúng một chỗ trong source code:

| Bước | File & hàm/class chính |
|---|---|
| CLI điều phối chung | `predict.py` → `src/pipeline_v2.py` (hàm `register_cli` — các lệnh `generate/repair/score/submit/check`) |
| [1] Dựng prompt | `src/executor.py` (`build_prompt`, `build_few_shot_block`); template s4fix: `runs/phase3/strategy_s4fix.json`; strategy retr: `src/retrieval.py` |
| [2] Sinh candidates | `src/pipeline_v2.py` (`GROUP_SPECS`, `cmd_generate`); client gọi server: `src/model.py` (`QwenInference.complete_many`) |
| [3] Extraction | `src/extraction_v2.py` (`extract_program_v2`, `_validate_program`) |
| [4] Repair | `src/pipeline_v2.py` (`cmd_repair`); prompt + schema: `src/extraction_v2.py` (`build_repair_message`, `REPAIR_SCHEMA`) |
| [5] Thực thi | `src/evaluator.py` (`evaluate_program_v2`, `flatten_nested_program`, `fuzzy_row_match`) |
| [6] Voting | `src/voting.py` (`cluster_and_vote`, class `VoteConfig`) |
| [7] Submission | `format_submission.py`; gate kiểm tra: `src/pipeline_v2.py` (`cmd_check`) |

Về hạ tầng, orchestration chạy local, inference chạy trên Cerebrium:

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

> 💡 **Vì sao tách đôi như vậy?** Máy local đóng vai nhạc trưởng: chuẩn bị câu hỏi, gửi đi, thu kết quả, bầu chọn — việc nhẹ, không cần GPU. Cerebrium là dịch vụ cloud cho thuê GPU: model 4 tỷ tham số cần GPU 24GB để chạy nhanh, thứ máy cá nhân không có. *Scale-to-zero* nghĩa là khi không có request, GPU tự tắt và không tính tiền — chỉ trả tiền đúng lúc đang suy luận.

#### 3.1.1 Pipeline dùng train / dev / test như thế nào?

Đề bài phát ba file dữ liệu trong `data/` (đọc và làm phẳng bởi `src/data.py`, hàm `_load_json_split`). Hệ thống cuối dùng mỗi tập cho một việc **hoàn toàn khác nhau**:

| Tập | Số câu | Có đáp án gold? | Vai trò trong hệ thống cuối |
|---|---|---|---|
| `train.json` | 2,986 | ✅ (câu hỏi + gold program) | **Kho bài mẫu**: nguồn few-shot cố định của strategy s4fix (kế thừa từ vòng tiến hóa Giai đoạn 2) và toàn bộ index TF-IDF của strategy retr (mục 3.3b). **Không dùng để train trọng số model** — hệ thống không fine-tune. |
| `dev.json` | 584 | ✅ | **Thước đo để ra quyết định**: mọi ablation A1–A5 (mục 5.2) chấm trên tập này; quyết định chọn A4 làm hệ thống cuối và loại judge đều dựa trên dev. |
| `test.json` | 494 | ❌ | **Chỉ chạy suy luận đúng một lượt** qua pipeline [1]→[7] để sinh `submission.csv` nộp Kaggle. Không có nhãn nên không thể (và không được) tune trên test. |

Lưu ý cho người mới: chữ "train" ở đây dễ gây hiểu lầm. Vì không fine-tune, `train.json` không hề "huấn luyện" model — nó chỉ là tập duy nhất có đáp án chuẩn, nên được dùng làm **ví dụ nhét vào prompt** (few-shot). Model học "trong lúc đọc đề", không học vào trọng số.

> 💡 **Vì sao phải tách 3 tập?** Nếu vừa chỉnh hệ thống vừa nhìn kết quả trên chính tập sẽ chấm điểm, ta sẽ vô thức "học thuộc" tập đó (overfit) — điểm đo được cao ảo, gặp câu mới là rơi. Quy tắc: nhìn đáp án dev bao nhiêu cũng được (nó sinh ra để thử nghiệm), nhưng tuyệt đối không dò đáp án test/leaderboard. Xem thêm mục 5.4.

### 3.2 Tận dụng kết quả Giai đoạn 2

> 💡 **EvoAgent "tự cải thiện" nghĩa là gì?** Một *strategy* đơn giản là một file JSON gồm: prompt template (lời dặn model) + danh sách few-shot. Vòng lặp EvoAgent cho LLM tự tiến hóa strategy qua 3 bước mỗi vòng: **propose** — LLM đọc strategy hiện tại và tự đề xuất bản cải tiến (`src/self_proposer.py`, hàm `propose_self`); **evaluate** — chấm strategy mới trên tập dev (`src/executor.py`, hàm `evaluate`); **reflect** — LLM đọc các câu sai và tự rút kinh nghiệm cho vòng sau (`src/self_reflector.py`, hàm `reflect_self`). Toàn bộ vòng lặp điều phối bởi `src/harness.py` (hàm `run_evoagent`). Con người không sửa tay prompt — model tự sửa cho chính nó.

Giai đoạn 3 kế thừa bốn tài sản cụ thể từ Giai đoạn 2:

1. **Strategy vô địch của vòng tiến hóa (`iter_004`).** Qua 5 vòng, dev accuracy đi 0.333 → 0.525 → 0.546 → 0.538 → **0.579** (ghi trong `evolution_proof.json`); strategy thắng cuộc nằm nguyên trong `runs/exp_self/iter_004_strategy.json`. Toàn bộ prompt template + bộ few-shot của nó được dùng làm gốc cho strategy `s4fix` (chỉ sửa 2 chỗ, xem khối (b) mục 3.3). Đây là kết quả trực tiếp, đo được, của quá trình tự cải thiện.
2. **Bộ few-shot hình thành và được sàng lọc tự động qua vòng tiến hóa.** Trong 4 bài mẫu của iter_004: #1–#2 đến từ strategy hạt giống (viết sẵn trong `make_seed_strategy`, `src/strategy.py`) và được kế thừa nguyên vẹn qua các vòng; #3–#4 do chính Qwen3.5-4B **tự viết ra** khi đề xuất strategy mới (hàm `propose_self`, `src/self_proposer.py`) — tự bịa passage mini, tự viết chương trình lẫn lời giải thích. Bài mẫu model đề xuất phải qua validator cú pháp `_is_valid_dsl_program` (chặn phép tính thô `=`, `+`, `*`, `/`; đòi có ít nhất một op DSL hợp lệ); vòng lặp còn một nhánh sinh bài mẫu từ câu train thật, với chuỗi CoT chỉ được giữ khi bản demo của model chốt lại **đúng nguyên văn** gold program (so khớp chuỗi sau chuẩn hóa — hàm `generate_few_shot_reasoning`). Lưu ý giới hạn: không cửa kiểm nào **thực thi** chương trình của bài mẫu — đây chính là kẽ hở để few-shot #4 dạng nested lọt lưới (khối (b) mục 3.3).

   > 💡 **CoT (Chain-of-Thought / chuỗi suy nghĩ)** — model được phép "nghĩ ra giấy" trước khi chốt đáp án: phần suy luận nằm giữa cặp thẻ `<think>...</think>`, phần đáp án nằm sau thẻ đóng. Nghĩ ra giấy giúp giải đúng bài nhiều bước, nhưng cũng sinh ra rủi ro mới — model có thể nghĩ mãi không dừng (xem "runaway" ở khối (c)).

3. **Hồ sơ bệnh án của baseline.** Gồm hai nguồn: (i) 240 raw output dev của iter_004 — artifact còn lưu nguyên tại `runs/exp_self/iter_004_eval_dev.json`, về sau được phân tích lại per-question trong `runs/phase3/rescore_report.json`; (ii) kết quả điều tra raw output test của lần submit baseline tại thời điểm đó (file trung gian không giữ lại trong repo, các con số dẫn ở mục 5.1). Mỗi khối của Giai đoạn 3 sinh ra để trị đúng một bệnh trong hồ sơ này:

   | Chẩn đoán từ điều tra baseline | Khối Giai đoạn 3 trị nó |
   |---|---|
   | Runaway CoT 21%, nhóm này đúng 0% | (c) decoding + (d) không mót rác + (e) repair |
   | Request fail bị nuốt thành 0.0 (9.3% test) | (h) reliability layer |
   | Extraction gom rác → crash evaluator | (d) extraction v2 |
   | Few-shot #4 dạy nested call không chạy được | (b) sửa few-shot + (f) flatten trong evaluator |
   | Lỗi dấu/chiều thay đổi, số kiểu Việt | (b) 2 hint mới trong prompt |

4. **Hạ tầng tái dùng.** App Cerebrium + vLLM giữ nguyên cấu hình serving (`cerebrium.toml`); executor DSL (`src/evaluator.py`) và bộ đếm chi phí `TokenBudget` (`src/executor.py`) của Giai đoạn 2 được nâng cấp thành evaluator v2 (khối (f)); 2,986 cặp (câu hỏi, gold program) trong `data/train.json` trở thành kho tra cứu few-shot cho strategy retrieval.

### 3.3 Triển khai Giai đoạn 3: từng khối theo thứ tự dòng chảy

**(a) Serving.** Một app Cerebrium duy nhất (GPU A10 24GB) chạy vLLM serve `QuantTrio/Qwen3.5-4B-AWQ`, expose API OpenAI-compatible, scale-to-zero khi rảnh. Cấu hình nằm trọn trong `cerebrium.toml`, entrypoint nguyên văn:

```
vllm serve QuantTrio/Qwen3.5-4B-AWQ --host 0.0.0.0 --port 8000
     --max-model-len 16384 --gpu-memory-utilization 0.90
```

Giữ nguyên từ Giai đoạn 2 — mọi cải thiện nằm ở phía client/pipeline (`src/model.py`, class `QwenInference`), không đổi model hay serving.

> 💡 **Ba thuật ngữ serving:** *vLLM* — phần mềm server chuyên phục vụ LLM hiệu năng cao; điểm quan trọng với pipeline này là nó cho phép một request sinh nhiều lời giải (`n>1`) mà chỉ đọc prompt một lần. *AWQ* — kỹ thuật nén trọng số model xuống ~4-bit, giúp model 4B tham số chỉ chiếm ~5.6GB và chạy vừa GPU 24GB. *OpenAI-compatible API* — server nhận request theo đúng format API của OpenAI, nên client chỉ cần thư viện `openai` chuẩn, trỏ URL về Cerebrium.

**(b) Dựng prompt — 2 strategies, cùng base template.**

**Strategy A (`s4fix`)** = `iter_004` kế thừa từ Giai đoạn 2 với đúng 2 sửa đổi (bản chốt: `runs/phase3/strategy_s4fix.json`, trường `metadata.extra.phase3` ghi chú nguyên văn nội dung sửa):

- few-shot #4 vốn dạy nested call `divide(subtract(500,400),400)` — chính evaluator của đề không chạy được — đổi thành dạng phẳng `subtract(500,400), divide(#0,400)`. Đối chiếu hai bản: bản nested nằm trong `runs/exp_self/iter_004_strategy.json`, bản phẳng trong `strategy_s4fix.json` (cùng passage "Lãi ròng năm 2022 là 500, năm 2021 là 400.");
- bổ sung 2 hint vào prompt template, trích nguyên văn từ `strategy_s4fix.json`:
  - *"LƯU Ý HƯỚNG THAY ĐỔI: 'thay đổi/tăng trưởng TỪ năm A ĐẾN năm B' = giá_trị_B − giá_trị_A (lấy giá trị sau trừ giá trị trước; kết quả có thể ÂM nếu giảm)"* — ví dụ câu PMI: `subtract(108.50, 100)` chứ không phải `subtract(100, 108.50)`;
  - *"LƯU Ý SỐ KIỂU VIỆT: dấu chấm trong số như '11.228 tỷ' thường là phân tách hàng nghìn (= 11228); hãy đối chiếu độ lớn với ngữ cảnh trước khi dùng"*.

**Strategy B (`retr`)** dùng **cùng template** nhưng thay few-shot cố định bằng **retrieval per-question** (toàn bộ trong `src/retrieval.py`): TF-IDF char n-gram (2,4) trên 2,986 câu train, sơ tuyển top-9 gần nhất rồi chọn 3 shot đa dạng op-signature. Cơ chế gồm bốn bước:

1. Mỗi câu hỏi được cắt thành mọi cụm 2–4 ký tự liên tiếp (char n-gram — "doanh thu" sinh `do`, `oa`, …, `doan`, `oanh`); so khớp ở mức ký tự nên chịu được biến thể chính tả tiếng Việt ("tỷ suất"/"tỉ suất" chỉ lệch đúng các cụm chứa `ỷ`/`ỉ`, phần lớn cụm còn lại vẫn trùng) mà không cần model embedding. Code: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=200_000)` trong `TrainIndex.build`.
2. TF-IDF đánh trọng số: cụm phổ biến ở mọi câu ("là bao", "nhiêu") gần 0, cụm hiếm mang nội dung ("khấu ha", "tỷ suấ") trọng số cao — mỗi câu thành một vector và độ giống giữa hai câu = độ trùng hai vector (cosine).
3. Xếp hạng cả 2,986 câu train theo độ giống với câu đang hỏi, giữ **9 câu đầu** làm sơ tuyển (code: `topk(question, k=max(8, n_shots*3))` với `n_shots=3` → k=9, trong hàm `make_few_shots`).
4. Prompt chỉ chứa được 3 shot, nhưng lấy thô top-3 thường được 3 bản sao cùng một khuôn tính — nên duyệt từ trên xuống và bỏ qua câu có **op-signature** (chuỗi tên phép tính của gold program, tính bởi hàm `_op_signature`; vd. `subtract(108.50,100), divide(#0,100)` → `subtract,divide`) trùng với shot đã chọn, để 3 shot cùng chủ đề nhưng trình diễn 3 khuôn giải khác nhau (nếu không gom đủ 3 signature khác nhau thì chấp nhận lấy trùng).

Minh họa thuật toán chọn ở bước 4 — duyệt top-9 từ trên xuống, câu nào có signature **chưa gặp** thì lấy, **gặp rồi** thì bỏ qua, đủ 3 thì dừng:

| Hạng | Op-signature | Quyết định |
|---|---|---|
| 1 | `subtract,divide` | ✅ lấy — shot 1 (khuôn mới) |
| 2 | `subtract,divide` | ⏭️ bỏ (trùng shot 1) |
| 3 | `subtract,divide` | ⏭️ bỏ (trùng) |
| 4 | `divide` | ✅ lấy — shot 2 (khuôn mới) |
| 5 | `subtract,divide` | ⏭️ bỏ (trùng) |
| 6 | `subtract,divide,multiply` | ✅ lấy — shot 3 (khuôn mới) → **dừng** |

Kết quả: 3 shot vẫn **cùng chủ đề** với câu đang hỏi (đều thuộc top-9 giống nhất) nhưng trình diễn **3 công thức giải khác nhau** — mỗi suất bài mẫu dạy được một điều mới, thay vì 3 bản sao của cùng một khuôn.

Lưu ý vai trò của từng loại dữ liệu: phép **so khớp chỉ dùng câu hỏi** (index TF-IDF xây trên 2,986 câu hỏi train — lúc suy luận, câu hỏi là thứ duy nhất ta có; câu test không có đáp án để so). **Gold program chỉ tham gia sau khi đã tìm xong** top-9, ở ba vai: bộ lọc đa dạng (op-signature ở bước 4), nội dung bài mẫu nhét vào prompt, và định vị cửa sổ ngữ cảnh ~500 ký tự quanh các con số của nó (hàm `_shot_passage`, hằng `MAX_PASSAGE_CHARS=500` — bài mẫu chỉ chứa đúng khúc văn bản liên quan thay vì cả trang báo cáo). Câu train không có gold program bị loại khỏi index từ đầu.

Ví dụ với câu PMI, retrieval trả về các câu train cùng dạng "tỷ lệ tăng trưởng… từ năm X đến năm Y" kèm gold program phẳng của chúng — few-shot chính là chương trình chuẩn thực thi được, không phải CoT tự sinh. Ensemble hai strategy chỉ khác nhau đúng một biến (nguồn few-shot) nên dễ quy kết nguyên nhân khi ablation.

**Tóm lại — ba "bản thể" của cùng một strategy.** Nhìn bằng diff sẽ thấy rõ mức độ kế thừa. Template: s4fix = iter_004 **cộng đúng 2 câu hint nối vào đuôi** (retr dùng y nguyên template của s4fix):

```diff
  Bạn là chuyên gia phân tích tài chính. Giải quyết bài toán theo trình tự chặt chẽ:
  1. Phân tích & Trích xuất... 2. Logic Toán học... 3. Chương trình: Chuyển đổi sang
  hàm riêng biệt. Tham chiếu kết quả trước bằng #0, #1...
  Quy tắc: Phần trăm phải là tỷ lệ thập phân (0.05), KHÔNG nhân 100. [...]
+ LƯU Ý HƯỚNG THAY ĐỔI: 'thay đổi/tăng trưởng TỪ năm A ĐẾN năm B' = giá_trị_B − giá_trị_A
+ (lấy giá trị sau trừ giá trị trước; kết quả có thể ÂM nếu giảm).
+ LƯU Ý SỐ KIỂU VIỆT: dấu chấm trong số như '11.228 tỷ' thường là phân tách hàng nghìn
+ (= 11228); hãy đối chiếu độ lớn với ngữ cảnh trước khi dùng.
```

Few-shot #4 (bài mẫu duy nhất bị sửa nội dung):

```diff
  passage: "Lãi ròng năm 2022 là 500, năm 2021 là 400."
- answer:  "divide(subtract(500, 400), 400)"      ← nested — evaluator của đề không chạy được
+ answer:  "subtract(500, 400), divide(#0, 400)"  ← phẳng — chạy được
```

Toàn cảnh ba bản thể:

| | `iter_004` (Giai đoạn 2) | Strategy A — `s4fix` | Strategy B — `retr` |
|---|---|---|---|
| File | `runs/exp_self/iter_004_strategy.json` | `runs/phase3/strategy_s4fix.json` | không có file riêng — nạp chung `strategy_s4fix.json` |
| Prompt template | bản gốc do vòng tiến hóa viết ra | = iter_004 **+ 2 hint** (diff trên) | = s4fix, y hệt |
| Few-shot | 4 bài **cố định**, #4 dạng nested | 4 bài cố định như iter_004, #4 đã flatten | **3 bài truy hồi riêng cho từng câu hỏi**, thay tại runtime (hàm `strategy_for_row`, `src/retrieval.py`) |
| Reasoning trong bài mẫu | #1–#2 từ seed viết sẵn, #3–#4 model tự viết khi propose (mục 3.2) | như iter_004 (#4 thêm câu nhắc "KHÔNG lồng hàm vào nhau") | 1 câu generic + gold program của câu train |
| Đổi theo từng câu hỏi? | không | không | có — mỗi câu test một bộ shot khác |

Tức iter_004 → s4fix chỉ là **vá lỗi tối thiểu** (2 sửa đổi, giữ nguyên "linh hồn" strategy mà vòng tiến hóa tìm ra); s4fix → retr **đổi đúng một biến** là nguồn few-shot — ba bản thể chung một khung xương prompt.

**(c) Sinh 16 candidates — decoding chống runaway.** Baseline Giai đoạn 2 chạy greedy temp=0 không repetition penalty → 21% output rơi vào vòng lặp suy nghĩ vô hạn (model lặp mãi "kiểm tra lại…" và không bao giờ đóng `</think>`), nhóm này đúng **0%**.

Giai đoạn 3: mỗi strategy sinh 1 greedy + 7 sampled. Cấu hình nằm nguyên văn trong `GROUP_SPECS` của `src/pipeline_v2.py`:

```python
GROUP_SPECS = {
    "greedy":  dict(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                    repetition_penalty=1.05, max_new_tokens=3072),
    "sampled": dict(temperature=0.6, top_p=0.95, top_k=20,
                    repetition_penalty=1.05, max_new_tokens=2048),
}
```

7 sampled gói trong **một request `n=7`** để chia sẻ prefill trên vLLM (prompt chỉ được đọc một lần cho cả 7 lời giải), chi phí gần bằng 1 request; seed sampling đặt cố định theo từng câu (`crc32(id)` trong `cmd_generate`) nên chạy lại vẫn tái lập đúng kết quả.

Tổng: 2 × (1+7) = **16 candidates/câu**; sampled tạo đa dạng cho voting, greedy làm mỏ neo ổn định.

> 💡 **Greedy vs sampled** — model sinh từng token một, mỗi bước có một bảng xác suất cho token kế tiếp. *Greedy* (temperature=0) luôn bốc token xác suất cao nhất → deterministic, cùng prompt luôn ra cùng kết quả — nhưng khi đã rơi vào vòng lặp thì không tự thoát được (vì thế cần *repetition penalty* — phạt nhẹ những token đã dùng để đỡ lặp). *Sampled* gieo token theo phân phối xác suất (điều tiết bởi temperature) → mỗi lần sinh đi một ngả suy luận khác nhau, thăm dò được các lối giải mà greedy bỏ lỡ — điều kiện cần để voting có ý nghĩa (8 lần temp=0 sẽ ra 8 bản y hệt).

> 💡 **3 few-shot ≠ 8 lần sinh** — hai trục độc lập: few-shot là số bài mẫu ở **đầu vào** (nằm trong prompt, dạy trước khi làm); 1+7 là số lời giải lấy ở **đầu ra** (cả 8 lần sinh dùng chung đúng một prompt, chỉ khác ngả suy luận).

**(d) Extraction v2 — bóc chương trình khỏi raw output** (`src/extraction_v2.py`, hàm `extract_program_v2`). Output tử tế có dạng:

```
<think>…100 → 108.50, tăng 8.50, chia gốc 100…</think>
{"Program syntax": "subtract(108.50, 100), divide(#0, 100)", "Numerical result": 0.085}
```

**Vì sao khâu này từng là điểm chết của baseline.** Bộ bóc cũ của Giai đoạn 2 (hàm `extract_answer` trong `src/model.py`) xử lý output tử tế thì ổn, nhưng khi output lộn xộn (runaway, không có JSON) nó rơi vào chế độ "mót": quét **từng dòng của toàn bộ đoạn độc thoại**, dòng nào chứa tên phép tính hoặc `#N` thì cắt lấy các cụm `op(...)`, rồi **nối tất cả mảnh mót được bằng dấu phẩy thành một "chương trình" duy nhất** (dòng code `", ".join(valid_parts)`). Điểm chết: trong bài toán tính toán, phần suy nghĩ nháp chứa đầy cụm `op(...)` — model vừa nghĩ vừa viết thử, kể cả các phương án nó đã tự bác bỏ — và bộ bóc cũ không phân biệt được "nháp" với "đáp án chốt". Ví dụ minh họa với câu PMI, một output runaway trông thế này:

```
Chênh lệch: subtract(108.50, 100) = 8.50. Tỷ lệ = divide(8.50, 100) = 0.085.
Hay là phải dùng divide(108.50, 100)? Không, đó là tỷ số chứ không phải tăng trưởng.
Thử lại: subtract(108.50, 100), rồi divide(#0, 100). Khoan, kiểm tra lại...
```

và thứ bộ bóc cũ mót ra không phải lời giải, mà là **bản ghi âm mọi phương án model từng cân nhắc** xếp cạnh nhau:

```
subtract(108.50, 100), divide(8.50, 100), divide(108.50, 100), subtract(108.50, 100), divide(#0, 100)
```

Chuỗi Frankenstein này giết evaluator cũ (`evaluate_program` trong `src/evaluator.py` — một parser chặt chẽ) theo đủ kiểu: mảnh mót bị cụt ngoặc hoặc dính chữ nghĩa → parse lỗi; `#ref` viết cho mạch suy nghĩ gốc nhưng sau khi nối lại theo thứ tự xuất hiện thì trỏ vào bước nháp khác hẳn; table op gọi tên hàng model bịa ra lúc nghĩ → không khớp hàng nào. Evaluator ném exception → câu đó 0 điểm: **47/240 câu dev (~20%) của baseline chết kiểu này** — mất điểm ở khâu bóc, bất kể suy luận đúng hay sai; oan nhất là như ví dụ trên, lời giải đúng `subtract(108.50, 100), divide(#0, 100)` *đã nằm sẵn trong nháp*. Và trường hợp chuỗi rác **chạy được** còn nguy hiểm hơn crash: nó ra một con số sai nghe hợp lý (chính là "phiếu độc" nói ở đoạn dưới).

Extraction v2 chọn triết lý ngược hẳn: không mót, chỉ nhận chương trình ở những vị trí đáng tin — ladder **4 nấc**, thử lần lượt từ đáng tin nhất xuống:

1. **JSON sau `</think>` cuối cùng** — đường lành mạnh, như ví dụ trên;
2. **JSON bị cắt cụt vì hết token — nhưng chuỗi chương trình đã kịp viết xong** (vd. `{"Program syntax": "subtract(108.50, 100), divide(#0, 100)", "Numerical re` — mất dấu `}` đóng khối nên parser JSON bó tay, song giá trị cần lấy vẫn nguyên vẹn) — bóc bằng regex bám theo key (`_KEY_PROGRAM_RE = r'"[Pp]rogram[ _]?syntax"\s*:\s*"([^"\n]+)"'`). Lưu ý regex đòi đủ **dấu nháy đóng** sau chương trình: nếu bản thân chương trình cũng bị chém giữa chừng (vd. `"subtract(108.50, 100), div`) thì nấc này không nhận — DSL cụt là vô dụng, sample rơi tiếp xuống nấc dưới rồi thường là repair;
3. **JSON hợp lệ ở bất kỳ đâu trong output** — vớt trường hợp model in JSON xong lại lan man tiếp phía sau, hoặc chốt đáp án ngay bên trong khối suy nghĩ;
4. **Quét từ dưới lên tìm dòng thuần-program** — dòng chỉ chứa tên op hợp lệ + số (hàm `_line_is_program_like`).

Minh họa hai nấc vét cuối (ví dụ giả định theo đúng dạng lỗi thực tế). **Nấc 3** cứu output chốt đáp án *bên trong* khối suy nghĩ — sau `</think>` không còn JSON nào (nấc 1, 2 đều trượt), nhưng khối JSON cân bằng vẫn nằm nguyên trong phần think:

```
<think>…Vậy chương trình là:
{"Program syntax": "subtract(108.50, 100), divide(#0, 100)", "Numerical result": 0.085}
Kiểm tra lại lần cuối cho chắc…</think>
Đáp án: tỷ lệ tăng trưởng là 8.5%.
```

**Nấc 4** cứu output không có JSON nào nhưng có dòng chốt thuần chương trình:

```
</think>
Ta lấy giá 2013 trừ giá 2012 rồi chia cho giá gốc.
Chương trình: subtract(108.50, 100), divide(#0, 100)
```

Quét từ dưới lên, nhãn `Chương trình:`/`PROGRAM:` được bỏ trước khi xét, phần còn lại chỉ gồm tên op hợp lệ + số → nhận. Chốt an toàn của nấc này là hàm `_line_is_program_like`: dòng trộn văn xuôi như *"Vậy ta tính subtract(108.50, 100) trước"* bị loại thẳng (các token "Vậy", "ta", "tính"… không phải op) — chỉ nhận **nguyên một dòng chương trình hoàn chỉnh**, không tái phạm kiểu mót từng mảnh của baseline.

Extraction chạy ở cấp **từng candidate** (16 lượt bóc/câu, mỗi raw output xử lý độc lập, mỗi candidate một dòng cache riêng); các bước (d)–(f) đều per-candidate, chỉ đến khối (g) voting 16 kết quả mới gộp lại thành một đáp án.

**Tuyệt đối không mót fragment từ output runaway** — output nào không có `</think>` bị trả rỗng ngay từ cửa (tag `runaway` trong code) và chuyển cho repair. Lý do: với voting, một phiếu **rỗng** vô hại (và còn được repair cứu), còn một chương trình rác-nhưng-chạy-được là một phiếu **độc** — nó thực thi ra một con số nghe hợp lý và kéo lệch cuộc bầu chọn. Thà phiếu trắng còn hơn phiếu giả.

Chương trình bóc ra còn phải qua 4 cửa validate cú pháp (hàm `_validate_program`), trượt cửa nào loại cửa đó:

1. **Op hợp lệ** — từng bước phải gọi hàm có trong DSL; hàm model bịa ra (vd. `percent(8.5, 100)` hay `sum(...)`) giữ lại chỉ để crash evaluator → loại.
2. **`#ref` đúng thứ tự** — `#N` chỉ được trỏ về bước đã tính trước đó; tham chiếu "tương lai" (vd. bước 2 gọi `#5`) không thể thực thi tuần tự → loại.
3. **≤ 8 bước** (`MAX_STEPS = 8`) — bài FinQA thật hiếm khi quá vài phép tính; "chương trình" dài hơn gần như chắc chắn là rác nối từ output lặp, không phải lời giải → loại.
4. **Dedupe bước lặp** — trường hợp cứu được thay vì loại: hai bước liền kề y hệt nhau (vd. `divide(#0, 100), divide(#0, 100)` — vết tích model lặp) thì gộp còn một.

**(e) Repair pass — cơ hội thứ hai** (`src/pipeline_v2.py`, hàm `cmd_repair`; prompt vá dựng bởi `build_repair_message` trong `src/extraction_v2.py`). Diện quét: mọi sample **không có chương trình thực thi được** — cụ thể là request fail, hoặc không bóc được chương trình (`program is None`), hoặc bóc được mà thực thi không ra giá trị (`value is None`). Năm nhóm nguyên nhân thực tế:

1. **Runaway — nhóm lớn nhất (~19.5% sample)**: model kẹt trong vòng lặp suy nghĩ, trần `max_new_tokens` chém đứt output khi vẫn còn trong `<think>` nên không tồn tại phần đáp án. Lưu ý: không phải "bài khó nên thiếu chỗ suy nghĩ" — model lặp vô hạn, cho thêm token cũng không dừng; trần chỉ là cái cắt cụt.
2. Đóng `</think>` tử tế nhưng **JSON đáp án bị cắt cụt quá sớm** — cũng chạm trần token nhưng ở pha viết đáp án; nếu chuỗi chương trình đã kịp viết xong thì nấc 2 của ladder cứu được bằng regex, cắt sớm hơn nữa mới rơi xuống repair.
3. Output hoàn chỉnh nhưng **chương trình trượt validate** (bịa hàm ngoài DSL, `#ref` sai thứ tự, >8 bước) hoặc trả lời văn xuôi/đáp án thẳng không kèm chương trình.
4. Chương trình **hợp lệ cú pháp nhưng thực thi lỗi** — table op không khớp được hàng nào (kể cả fuzzy match), chia cho 0.
5. **Request fail hẳn** — hiếm sau reliability layer, quét cho đủ.

Cách vá: đúng 1 lần re-ask — gửi lại **1200 ký tự cuối** của bản nháp lỗi (code: `raw_tail[-1200:]`), ép trả lời bằng **guided-JSON decoding** theo schema `REPAIR_SCHEMA = {"Program syntax": str}`, cap 256 tokens, temperature 0. Một lượt repair hoàn chỉnh cho câu PMI trông như sau — đúng cấu trúc prompt mà `build_repair_message` dựng (các câu chữ dặn dò là nguyên văn từ code; bản nháp là giả định theo dạng lỗi thực tế):

```
[System]  Bạn là trợ lý AI chuyên phân tích tài chính tiếng Việt. Nhiệm vụ: viết lại
          MỘT chương trình DSL phẳng, đúng cú pháp, trả lời câu hỏi dựa trên ngữ cảnh.

[User]    Bối cảnh: …bảng giá cổ phiếu PMI… 31/12/2012: $100.00 … 31/12/2013: $108.50…

          Câu hỏi: Tỷ lệ tăng trưởng giá cổ phiếu của PMI từ năm 2012 đến 2013 là bao nhiêu?

          Bản nháp trước (có thể sai cú pháp hoặc bị cắt giữa chừng — chỉ dùng tham khảo):
          …108.50 − 100 = 8.50, chia gốc 100 ra 0.085. Khoan, kiểm tra lại: 108.50 −

          [9 quy tắc DSL — khối _DSL_BLOCK dùng chung với prompt generation]

          Hãy trả về DUY NHẤT một khối JSON dạng {"Program syntax": "<chương trình phẳng>"}.
          Chương trình phải phẳng (KHÔNG lồng hàm), các bước phân cách bằng dấu phẩy,
          tham chiếu bước trước bằng #0, #1...

[Model]   {"Program syntax": "subtract(108.50, 100), divide(#0, 100)"}   ← guided-JSON, ≤256 token
```

Bản nháp đuôi 1200 ký tự chính là "phần đã làm" của model; guided-JSON đóng vai trò tờ phiếu trả lời chỉ có đúng một ô trống.

> 💡 **Guided-JSON decoding là gì?** Bình thường model muốn sinh token gì cũng được. Ở chế độ guided-JSON, server vLLM **chặn ngay tại bước sinh token** mọi token làm output lệch khỏi schema JSON đã khai — model như bị ép điền vào form có sẵn, muốn lan man cũng không sinh nổi ký tự ngoài khuôn `{"Program syntax": "..."}`.

Thiết kế nhắm thẳng nhóm 1: trong văn bản runaway, model thường **đã tính ra đáp án** — nó chỉ không chịu dừng để chốt; repair tước quyền suy nghĩ tiếp (256 token không đủ chỗ lan man), chỉ cho phép chốt chương trình từ những gì đã nháp. Vì vậy vá thành công 660/987 sample hỏng chỉ trong 6.5 phút — không giải lại bài, chỉ "bắt nộp phần đã làm".

**(f) Evaluator v2 — thực thi chương trình ra giá trị số** (`src/evaluator.py`). Nâng cấp executor của Giai đoạn 2 (hàm `evaluate_program` giữ nguyên làm tham chiếu; bản mới là `evaluate_program_v2`) với ba sửa đổi:

1. Tự động **flatten nested call** (hàm `flatten_nested_program`) — nếu model vẫn viết `divide(subtract(500, 400), 400)` thì chuyển thành `subtract(500, 400), divide(#0, 400)` thay vì crash (đây chính là ví dụ trong docstring của hàm).
2. **Fuzzy row match** cho table ops (hàm `fuzzy_row_match`: thử khớp chính xác → chứa chuỗi → bỏ dấu/hoa-thường/hậu tố đơn vị → Jaccard ≥ 0.6) — ví dụ chương trình gọi hàng "P/E" nay khớp được header thật "P/E (x)".

   > 💡 **Jaccard** — độ đo trùng lặp giữa hai tập từ: (số từ chung) ÷ (số từ của cả hai gộp lại). "P/E" và "P/E (x)" chung 1 từ trên tổng 2 từ → 0.5… cộng thêm các nấc chuẩn hóa phía trước nên vẫn khớp; ngưỡng 0.6 đủ chặt để không khớp bừa hai hàng khác nghĩa.

3. **Chuẩn hóa số âm kế toán** (hàm `extract_numbers_v2`) — ô `"$ -61.1 ( 61.1 )"` đọc thành −61.1 (báo cáo tài chính hay ghi số âm kèm bản tuyệt đối trong ngoặc; bug gốc đọc thành hai số khiến `table_max` chọn sai dấu — chính doc-test của `evaluator.py` đề phát cũng sai vì bug này).

**(g) Voting self-consistency — chọn giá trị cuối** (`src/voting.py`, hàm `cluster_and_vote`). Cả 16 candidates được thực thi hết và **vote theo GIÁ TRỊ đã execute**, không vote theo text (hai chương trình viết khác nhau nhưng cùng ra 0.085 vẫn là một phiếu chung).

> 💡 **Self-consistency** — thay vì tin một lời giải duy nhất, sinh nhiều lời giải độc lập rồi lấy đáp án xuất hiện nhiều nhất. Trực giác: có nhiều đường suy luận sai, nhưng chúng sai *khác nhau*; các đường đúng lại *hội tụ về cùng một con số*. Đáp án được nhiều lời giải độc lập cùng chạm tới đáng tin hơn hẳn đáp án của một lời giải đơn lẻ.

Các giá trị được cluster single-link với tolerance `max(1e-4, 0.5% × max(|a|,|b|))` (hàm `_tolerance`; a, b là hai giá trị đang so) — 0.085 và 0.08498 vào chung cluster.

Tolerance cần thiết vì số thực hiếm khi bằng nhau tuyệt đối (làm tròn khác nhau ở bước trung gian); so bằng `==` sẽ xé vote thành 16 phe lẻ. Nó là **dung sai kép** — lấy vế lớn hơn, mỗi vế cứu một thái cực:

- Sàn tuyệt đối `1e-4` cứu các **đáp án nhỏ** quanh 0: với đáp án 0.001, dung sai tương đối 0.5% chỉ là 5·10⁻⁶ — khắt khe phi lý so với sai số làm tròn.
- Vế tương đối `0.5%` cứu các **đáp án lớn**: hai lời giải ra 11,228 và 11,230 rõ ràng cùng đáp án, nhưng dung sai 1e-4 sẽ tách chúng; 0.5% của 11,228 ≈ 56 mới đúng thang đo.

Bản chất: dung sai **tự co giãn theo thang đo của đáp án** — đáp án dạng tỷ lệ thì khắt khe cỡ phần vạn, đáp án dạng nghìn tỷ thì nới ra vài chục.

Ví dụ minh họa (giả định) cho câu PMI: 11 candidates ra ≈ 0.085 (gồm cả 2 greedy), 2 ra 8.5 (quên chia 100), 1 ra 0.185, 2 hỏng không thực thi được → cluster 0.085 thắng áp đảo.

Trọng số phiếu và ngưỡng chấp nhận là hằng số khai trong class `VoteConfig`:

```python
abs_tol = 1e-4;  rel_tol = 0.005          # dung sai kép
greedy_weight_bonus = 0.5                  # phiếu greedy nặng hơn
stated_agree_bonus = 0.25                  # thưởng khi execute khớp số model tự khai
accept_confidence = 0.5;  accept_min_members = 3;  accept_margin = 0.15
```

Tức mỗi phiếu nặng 1.0, +0.5 nếu là greedy, +0.25 nếu giá trị execute khớp "Numerical result" model tự khai trong JSON; giá trị đại diện = của greedy member trong cluster thắng (nếu có).

Ba ngưỡng đồng thuận (tự đặt, không học từ dữ liệu) đo ba khía cạnh khác nhau của một chiến thắng đáng tin:

| Ngưỡng | Công thức | Ý nghĩa |
|---|---|---|
| confidence ≥ 0.5 | trọng số cluster thắng ÷ tổng trọng số mọi phiếu | thắng phải **quá bán** — chiếm 30% trong khi 70% còn lại rải rác thì chưa tin được |
| members ≥ 3 | số phiếu trong cluster thắng | ít nhất 3 lời giải độc lập cùng ra một số — 1-2 phiếu trùng nhau có thể chỉ là ăn may |
| margin ≥ 0.15 | (cluster nhất − cluster nhì) ÷ tổng | thắng phải **cách biệt** — nhất 40% nhì 38% là hai đáp án đang giằng co |

Lưu ý ngữ nghĩa: trượt ngưỡng **không làm mất đáp án** — vote vẫn trả về giá trị cluster thắng; ba ngưỡng chỉ gắn cờ `needs_judge` (kèm lý do `low_confidence` / `few_members` / `low_margin`) để quyết định câu nào đáng gửi trọng tài trong thử nghiệm A5. Ở hệ thống cuối A4 (không judge), cờ này không kích hoạt gì — mọi câu đều dùng đáp án cluster thắng.

**(h) Reliability layer — xuyên suốt mọi request.** Điều tra baseline phát hiện **46/494 câu test (9.3%) có raw output RỖNG** — request fail lúc pod restart bị nuốt im lặng thành 0.0. Giai đoạn 3 xử lý bằng hai lớp:

- **Retry exponential backoff + jitter** (`src/model.py`, hàm `QwenInference.complete_many`): phân loại lỗi transient (mã `{408, 409, 429, 500, 502, 503, 529}` + timeout — thử lại tối đa 5 lần) vs permanent (4xx khác — dừng ngay); công thức chờ giữa hai lần thử: `min(60, 2^attempt) × uniform(0.5, 1.5)` giây. Hết 5 lần vẫn lỗi thì **raise exception tường minh** (`TransientExhausted`) chứ không bao giờ trả chuỗi rỗng im lặng như baseline.

  > 💡 **Exponential backoff + jitter** — lỗi mạng/quá tải thường tự hết sau vài giây; chiến lược là chờ 2s, 4s, 8s… (lũy thừa — cho server thời gian hồi phục) và nhân thêm hệ số ngẫu nhiên 0.5–1.5 (*jitter*) để 24 request cùng fail không đồng loạt retry vào đúng một thời điểm.

- **Scheduler `as_completed` + cache JSONL** (`src/pipeline_v2.py`, class `PredictionCache`): kết quả nào xong trước ghi ngay xuống đĩa dòng đó, mỗi dòng một sample với key `(qid, strategy, sample_index)` + status. Một dòng cache trông như sau (minh họa rút gọn theo đúng schema `GenRecord`):

```json
{"qid": "PM/2017/page_25.pdf-1", "strategy_id": "s4fix", "sample_index": 0,
 "status": "ok", "program": "subtract(108.50, 100), divide(#0, 100)",
 "value": 0.085, "extraction_tag": "json_after_think", "raw_output": "<think>…", …}
```

Đứt mạng hay tắt terminal giữa chừng → chạy lại **đúng lệnh cũ**, cache tự resume phần thiếu (sample nào đã `ok`/`repaired` thì bỏ qua), không tốn lại GPU. Kết quả: **0/4,672 request thất bại** trên dev, 0 câu bị 0.0 oan trên test.

**(i) Post-processing & submission gate.** Giá trị cuối của 494 câu qua `format_submission.py` (dùng nguyên trạng của đề — giữ đúng thứ tự row, điền 0.0 câu thiếu); sau đó gate `predict.py check` (hàm `cmd_check` trong `src/pipeline_v2.py`) phải PASS trước khi upload:

- đúng 3 cột `id,Usage,predicted_value`, đủ 494 id đúng thứ tự;
- số câu 0.0 ≤ 10 (baseline có tới 102!);
- không scientific notation/−0.0;
- log thống kê hình dạng giá trị (số câu 0.0, số câu âm, số giá trị dạng tỷ lệ trong [−1, 1]) để soát nhanh bằng mắt trước khi upload.

Chỉ upload Kaggle một lần với file đã qua gate.

### 3.4 Những gì KHÔNG dùng trong hệ thống cuối

- **Fine-tuning**: không dùng (giữ nguyên serving config; xem mục 7).
- **LLM-as-judge**: đã thử nghiệm đầy đủ với DeepSeek-R1-Distill-Qwen-7B (`src/judge.py`) nhưng **loại khỏi pipeline cuối** vì ablation trên dev cho kết quả âm (mục 5.3).

---

## 4. Dữ liệu ngoài & dữ liệu synthetic

- **Dữ liệu ngoài: KHÔNG sử dụng.** Toàn bộ few-shot (cả cố định lẫn retrieval) lấy từ `data/train.json` được cung cấp; không crawl, không dataset bổ sung, không tra cứu/tái tạo nhãn test dưới bất kỳ hình thức nào.
- **Dữ liệu synthetic**: hai bài mẫu few-shot #3–#4 của strategy s4fix (passage mini + chương trình + lời giải) do chính pipeline EvoAgent (Qwen3.5-4B) tự sinh ở Giai đoạn 2 khi đề xuất strategy (`propose_self`, `src/self_proposer.py`), sàng lọc bằng validator cú pháp DSL; hai bài mẫu còn lại thuộc strategy hạt giống viết sẵn trong code (mục 3.2). Không dùng model ngoài để sinh nhãn hay pseudo-label.
- **Model sử dụng**: `QuantTrio/Qwen3.5-4B-AWQ` (4B, model chính — toàn bộ generation/repair) và `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7B, self-host thử nghiệm judge — không có trong pipeline cuối). Cả hai ≤ 9B, tuân thủ luật cuộc thi; không dùng API model đóng nào.

---

## 5. Thí nghiệm, Ablation & Lựa chọn hệ thống cuối

### 5.1 Điều tra lỗi baseline (động lực thiết kế)

Phân tích per-question trên output của baseline xác định điểm rơi. Hai nguồn số liệu: cột dev từ 240 raw output còn lưu trong `runs/exp_self/iter_004_eval_dev.json` (phân tích lại per-question trong `runs/phase3/rescore_report.json`); cột test từ điều tra raw output của lần submit baseline tại thời điểm đó (file trung gian không giữ lại trong repo):

| Lỗi hệ thống | Bằng chứng | Tỷ trọng |
|---|---|---|
| Runaway CoT (không đóng `</think>`) | 51/240 dev → **0% đúng** | ~50% tổng lỗi dev |
| Request fail bị nuốt thành 0.0 | 46/494 test raw output rỗng | 9.3% test |
| Extraction gom rác → crash evaluator | 47/240 dev | ~46% lỗi dev |
| Few-shot #4 dạy nested call | 28/494 test program nested → 0.0 | 5.7% test |
| Lỗi dấu/chiều thay đổi | 7/240 dev có pred = −gold | ~3% |

Số liệu dev-240 kiểm chứng lại được bằng chính extraction v2: chạy `predict.py rescore` trên 240 raw output cũ cho phân bố tag `json_after_think` 178 · `runaway` 51 · `json_truncated` 9 · `none` 2 (ghi trong `rescore_report.json`).

Phát hiện then chốt: trong các output đóng `</think>` tử tế, **100% chứa khối JSON parse được** → chỉ cần diệt runaway + extraction đúng chỗ là giải quyết phần lớn.

### 5.2 Bảng ablation chính (full dev-584, scorer abs ≤ 1e-4)

> 💡 **Ablation là gì?** Cách đo đóng góp của từng bộ phận: bật dần từng khối lên (hoặc tắt đi) rồi đo lại accuracy trên cùng một tập dev. Chênh lệch giữa hai dòng liền nhau = công của đúng khối vừa thêm — tránh ngộ nhận "hệ thống tốt lên nhờ X" trong khi thật ra nhờ Y.

| # | Cấu hình (cộng dồn) | Dev acc | Δ |
|---|---|---|---|
| — | Baseline iter_004 (dev-240, tham chiếu) | 0.5792 | — |
| A1 | Greedy + toàn bộ fixes (decoding, extraction v2, evaluator v2, s4fix, repair) | **0.6404** | +6.1 pp |
| A3 | + Self-consistency K=8, vote theo giá trị | **0.7106** | +7.0 pp |
| A4 | + Ensemble strategy retrieval (16 candidates) — **HỆ THỐNG CUỐI** | **0.7346** | +2.4 pp |
| A5 | + LLM-judge R1-Distill-7B cho câu bất đồng thuận | 0.7295 | **−0.5 pp** (loại) |

Số liệu sinh bởi `predict.py score --ablation A1/A3/A4` và lưu tại `runs/phase3/ablation_report.json` (kèm file eval per-question `ablation_A*_eval.json` cho từng cấu hình).

*Vì sao bảng nhảy từ A1 sang A3 (không có A2)?* Slot `A2` có tồn tại trong dict `ABLATIONS` của `src/pipeline_v2.py`, nhưng được cấu hình **y hệt A1** (`strategies=["s4fix"], greedy_only=True`) — chấm nó sẽ ra đúng con số của A1. Nguồn gốc: kế hoạch ban đầu dành sẵn hai mã A1/A2 cho chặng "greedy + repair", nhưng khi triển khai, repair ghi đè trạng thái ngay trên từng dòng cache và scorer nhận cả sample `ok` lẫn `repaired`, nên không tồn tại hai mốc đo tách biệt — hai mã chập làm một. A2 vì thế không được chấm, không có trong `ablation_report.json`; bảng giữ nguyên mã số gốc (A1/A3/A4/A5) để khớp artifacts.

Accuracy theo loại câu hỏi (A4 vs baseline): `table_op` 0.33 → **0.71**; `division` 0.68 → 0.79; `addition` 0.39 → 0.65 (vẫn yếu nhất); `subtraction` 0.62 → 0.73; `multiplication` 0.60 → 0.80.

### 5.3 Ablation âm: LLM-as-judge (bằng chứng phương pháp luận)

> 💡 **LLM-as-judge** — dùng một LLM thứ hai làm "trọng tài": đưa cho nó câu hỏi + các đáp án đang tranh chấp, nhờ nó phán đáp án nào đúng. Ý tưởng phổ biến, nhưng chỉ đáng tin khi trọng tài giỏi hơn hẳn thí sinh — điều thử nghiệm dưới đây cho thấy không xảy ra với model 7B.

Thử nghiệm (`src/judge.py`, hàm `run_judge`): các câu vote không đạt ngưỡng đồng thuận (~20-35%, cờ `needs_judge` từ khối (g)) được đưa cho DeepSeek-R1-Distill-Qwen-7B (self-host trên cùng app Cerebrium, swap model tuần tự) phán xử. Thiết kế tôn trọng đặc thù reasoning model:

- **Không chặn thẻ `<think>`** (không ép JSON từ token đầu — làm vậy giảm mạnh chất lượng suy luận).
- Verdict bóc bằng regex sau `</think>` (ba trường `SELECTION/PROGRAM/FINAL_VALUE`).
- Chỉ tin chương trình judge viết lại **sau khi thực thi local thành công**.

Kết quả trên dev: judge override 33 câu, làm **giảm** 3 câu ròng (0.7346 → 0.7295) — model 7B không đủ tin cậy để thắng majority vote của 16 candidates trên miền tài chính tiếng Việt. **Quyết định: loại judge khỏi hệ thống cuối, giữ A4.**

Bằng chứng lưu tại `runs/phase3/judge_cache.jsonl` (60 verdict, mỗi verdict kèm 2,000 ký tự đầu chuỗi suy nghĩ của judge).

### 5.4 Phương pháp validate

Toàn bộ quyết định dựa trên **full dev 584** (baseline chỉ dùng slice 240) — đúng phân vai ba tập dữ liệu đã nêu ở mục 3.1.1; test không có nhãn nên tuyệt đối không tune trên test; chỉ submit Kaggle 2 lần (baseline + final), không probe leaderboard.

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

Mọi subcommand ở trên được định nghĩa trong `src/pipeline_v2.py` (hàm `register_cli`). Mọi lệnh idempotent (cache JSONL tự resume); log tự ghi vào `runs/phase3/run.log`, tham số snapshot vào `phase3_config.json`.

### 6.3 Cấu hình model & sampling

| Tham số | Greedy | Sampled (n=7) | Repair | Judge (thử nghiệm) |
|---|---|---|---|---|
| temperature | 0.0 | 0.6 | 0.0 | 0.6 |
| top_p / top_k | 1.0 / — | 0.95 / 20 | 1.0 / — | 0.95 / — |
| repetition_penalty | 1.05 | 1.05 | 1.0 | 1.0 |
| max_new_tokens | 3072 | 2048 | 256 (guided JSON) | 4096 |

Vote: tolerance `max(1e-4, 0.5% × max(|a|,|b|))`, ngưỡng chấp nhận confidence 0.5 / members 3 / margin 0.15 (class `VoteConfig`, `src/voting.py`).

### 6.4 Compute, chi phí & commit

| Hạng mục | Input tokens | Output tokens |
|---|---|---|
| Generation (dev + test, Qwen) | 55,941,840 | 17,239,308 |
| Repair (Qwen, guided) | 8,792,622 | 170,416 |
| Judge thử nghiệm (R1-7B) | 118,240 | 41,844 |
| **Tổng** | **~64.9M** | **~17.5M** |

(Số liệu ghi tự động vào `runs/phase3/token_usage.json`.)

- GPU: ~14-15 giờ A10 (Cerebrium, scale-to-zero khi rảnh). Chi phí thực tế: 【Điền: tổng $ từ Cerebrium Dashboard → Billing, ước tính ~$15-20】.
- Không phát sinh chi phí API ngoài (judge tự host).
- Commit hash: 【Điền: hash sau commit cuối — tại thời điểm draft là `6df8af0657b1c629013954bb80a133d2a0a3318f`, cần commit nốt các thay đổi pipeline Giai đoạn 3 rồi cập nhật】.

---

## 7. Hạn chế & Bài học

**Hạn chế:**
1. **Runaway CoT chưa trị tận gốc** — repetition penalty 1.05 giảm nhưng vẫn ~19.5% sample runaway; hệ thống bù bằng vote + repair thay vì sửa tận gốc (cần fine-tune hoặc model mới hơn).
2. **`addition` vẫn yếu nhất (0.65)** — thực chất là nhóm câu đa bước phức tạp bị classifier gán nhãn addition; cần phân loại câu hỏi tốt hơn để nhắm few-shot chính xác.
3. **Dev–LB lệch +4.5pp** (0.735 vs 0.779) — scorer nội bộ abs ≤ 1e-4 khắt khe hơn metric Kaggle; may mắn lệch theo hướng có lợi nhưng làm giảm độ chính xác của dự báo dev.
4. **Judge 7B thất bại** — với budget model nhỏ, verifier không thắng được self-consistency; muốn judge hiệu quả cần model lớn hơn đáng kể hoặc verifier được huấn luyện chuyên biệt.
5. Chưa thử fine-tuning (ràng buộc giữ nguyên serving + thời gian) — hướng tiềm năng nhất còn lại.

**Bài học lớn nhất:** phần lớn điểm cải thiện (+11 trong +18 pp) đến từ **kỹ nghệ hệ thống** — điều tra per-question để tìm đúng chỗ rơi điểm (request bị nuốt, extraction gom rác, few-shot dạy sai, decoding sai chế độ) — chứ không phải từ model thông minh hơn.

"Đo trước, sửa đúng chỗ, mỗi thay đổi một ablation" hiệu quả hơn nhiều so với đổi kiến trúc theo cảm tính; và một ablation âm được đo nghiêm túc (judge) cũng giá trị không kém ablation dương.

---

## Phụ lục A — Bằng chứng đính kèm gói ThinkFlic

| File | Nguồn |
|---|---|
| `evidence/evolution_proof.json` | `./evolution_proof.json` (Giai đoạn 2, nguyên trạng) |
| `evidence/failure_mode_report.pdf` | `./runs/exp_self/failure_mode_report.pdf` |
| `evidence/learning_curve.pdf` | `./runs/exp_self/learning_curve.pdf` |
| `evidence/strategy_diversity.pdf` | `./runs/exp_self/strategy_diversity.pdf` |
| (bổ sung Phase 3) `runs/phase3/ablation_report.json`, `token_usage.json`, `run.log` | sinh tự động bởi pipeline |

## Phụ lục B — Bản đồ codebase (điểm vào cho người muốn đọc code)

Đường dẫn tính từ thư mục gốc `assignment03/`.

| File / thư mục | Vai trò |
|---|---|
| `predict.py` | Cổng CLI duy nhất của Giai đoạn 3; các subcommand đăng ký từ `src/pipeline_v2.py` |
| `src/pipeline_v2.py` | Trái tim Giai đoạn 3: `generate/repair/score/judge/submit/check`, cache `PredictionCache`, cấu hình decoding `GROUP_SPECS` |
| `src/retrieval.py` | Strategy `retr`: index TF-IDF trên train, chọn 3 few-shot đa dạng op-signature |
| `src/extraction_v2.py` | Bóc chương trình khỏi raw output (ladder 4 nấc) + 4 cửa validate + prompt/schema cho repair |
| `src/evaluator.py` | Thực thi DSL: bản Giai đoạn 2 (`evaluate_program`) + bản cứng hóa v2 (flatten, fuzzy match, số âm kế toán) |
| `src/voting.py` | Self-consistency voting: `cluster_and_vote`, ngưỡng trong `VoteConfig` |
| `src/judge.py` | Thử nghiệm LLM-as-judge với R1-Distill-7B (đã loại khỏi hệ thống cuối) |
| `src/model.py` | Client HTTP gọi vLLM/Cerebrium: `QwenInference.complete_many`, retry + backoff; kèm bộ bóc cũ `extract_answer` của Giai đoạn 2 (bài học dẫn tới extraction v2) |
| `src/executor.py` | Giai đoạn 2: dựng prompt (`build_prompt`, `_DSL_BLOCK`), chấm strategy (`evaluate`), bộ đếm `TokenBudget` |
| `src/harness.py` · `src/self_proposer.py` · `src/self_reflector.py` · `src/strategy.py` | Vòng tiến hóa EvoAgent Giai đoạn 2: orchestrator `run_evoagent`, propose → evaluate → reflect, định nghĩa `Strategy` |
| `src/data.py` | Đọc & làm phẳng `data/{train,dev,test}.json` |
| `format_submission.py` | Xuất `submission.csv` đúng format Kaggle |
| `cerebrium.toml` | Cấu hình serving vLLM trên Cerebrium (GPU A10, scale-to-zero) |
| `data/` | `train.json` (2,986 câu, có gold) · `dev.json` (584, có gold) · `test.json` (494, không gold) |
| `runs/exp_self/` | Artifacts Giai đoạn 2: strategy/eval iter_000–004, `evolution_proof.json`, 3 PDF phân tích |
| `runs/phase3/` | Artifacts Giai đoạn 3: cache dev/test, `ablation_report.json`, `token_usage.json`, `judge_cache.jsonl`, `strategy_s4fix.json`, `rescore_report.json` |
