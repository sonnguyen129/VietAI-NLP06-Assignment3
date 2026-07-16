# EvoAgent — Vietnamese FinQA (VietAI Advanced NLP 06 · Assignment 3)

Agent **tự tiến hóa prompt** (self-improving loop) cho bài toán hỏi đáp tài chính tiếng Việt: đọc báo cáo tài chính, sinh chương trình **FinQA DSL** để tính toán đáp án. Agent tự chấm điểm, tự phân tích lỗi rồi tự viết lại prompt của chính nó qua từng vòng lặp. Inference dùng `Qwen3.5-4B-AWQ` serve bằng vLLM trên **Cerebrium** (GPU A10); toàn bộ vòng lặp tiến hóa và graders chạy local.

**Kết quả Kaggle (hidden test):** baseline LB **0.59919** → final LB **0.77935** (self-consistency K=8 + ensemble 2 strategies; chi tiết ablation A1→A5 trong report).

## Các file quan trọng cần xem

| File | Nội dung |
|---|---|
| [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) | Đề bài gốc + hướng dẫn setup Cerebrium/môi trường chi tiết |
| [report/technical_report_draft.md](report/technical_report_draft.md) | Báo cáo kỹ thuật: phương pháp, ablation, phân tích lỗi |
| [src/harness.py](src/harness.py) | Vòng lặp tiến hóa + AFO parent selection (Stage 4) |
| [src/self_reflector.py](src/self_reflector.py) · [src/self_proposer.py](src/self_proposer.py) | Tự phân tích lỗi (Stage 2) & tự viết lại prompt (Stage 3) |
| [src/executor.py](src/executor.py) | Evaluation engine + token budget (Stage 1) |
| [src/pipeline_v2.py](src/pipeline_v2.py) | Pipeline Phase 3: self-consistency, ensemble, voting |
| [src/model.py](src/model.py) · [src/evaluator.py](src/evaluator.py) | HTTP client Cerebrium/vLLM · thực thi FinQA DSL |
| [main.py](main.py) · [run_cerebrium.py](run_cerebrium.py) · [submit.py](submit.py) | Entry points: CLI, job runner, tạo submission |
| [graders/](graders/) | 6 bộ chấm tự động theo từng stage |
| [runs/](runs/) · `*_proof.json` | Bằng chứng thí nghiệm: learning curve, ablation evals, token usage |
| [docs/](docs/) | Luật Kaggle (`PHASE3_KAGGLE.md`) & cấu trúc nộp bài (`THINKFLIC_SUBMISSION.md`) |
| [tutorial/](tutorial/) | Tổng quan project cho người mới + hướng dẫn vận hành Giai đoạn 2 |

## Gói nộp bài (submission package)

Gói ThinkFlic đầy đủ (report PDF, evidence, source code đóng gói, video) được lưu tại: **【Điền link submission】**
