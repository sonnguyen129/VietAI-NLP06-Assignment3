"""
self_proposer.py — Self-optimization proposer: uses Qwen itself as the meta-agent.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional, List
from pydantic import BaseModel, Field
from datasets import Dataset

from src.model import extract_answer
from src.executor import normalize_program, classify_question_type
from src.strategy import (
    CoTFormat,
    FewShotExample,
    RetrievalConfig,
    Strategy,
    StrategyHistory,
    StrategyMetadata,
)

class FewShotExampleSchema(BaseModel):
    passage: str
    question: str
    answer: str
    reasoning: Optional[str] = None

class ProposerSchema(BaseModel):
    hypothesis: str = Field(..., description="A short one-sentence hypothesis")
    instruction_phrasing: str = Field(..., description="General instructions / role / phrasing prefix for the model, without any placeholders")
    cot_format: str = Field(..., description="Must be 'none', 'stepbystep', or 'chain'")
    few_shot_examples: List[FewShotExampleSchema]
    reasoning: str = Field(..., description="A short one-sentence reasoning")

logger = logging.getLogger(__name__)

_VALID_COT = {f.value for f in CoTFormat}

_SYSTEM_PROPOSE = """\
Bạn là trợ lý nghiên cứu NLP đang thiết kế một chiến lược prompting \
để giúp một mô hình ngôn ngữ giải bài tập toán tài chính tiếng Việt (cộng, trừ, nhân, chia, đọc bảng).

Nhiệm vụ của bạn là đưa ra một chiến lược prompting mới dựa trên lịch sử các chiến lược đã thử và kết quả phản ánh gần nhất.

LƯU Ý QUAN TRỌNG VỀ CÚ PHÁP CHƯƠNG TRÌNH (BẮT BUỘC TUÂN THỦ TRONG FEW-SHOT EXAMPLES):
1. Mỗi bước là một hàm riêng biệt, phân cách bằng dấu phẩy
2. KHÔNG lồng hàm vào nhau (không dùng divide(table_average(...), ...))
3. Dùng #0, #1, #2... để tham chiếu kết quả của bước trước (bắt đầu từ #0)
4. Tên cột/hàng trong table_xxx KHÔNG dùng dấu ngoặc kép
5. table_xxx chỉ nhận đúng 2 tham số: (tên_hàng, none)
6. Số âm viết trực tiếp: add(-167.4, -53.3) — không dùng ngoặc thêm
7. CỰC KỲ QUAN TRỌNG VỀ TỶ LỆ PHẦN TRĂM: Kết quả đầu ra của chương trình PHẢI luôn ở dạng tỷ lệ thập phân (ví dụ: 0.05 thay vì 5%, hay 0.03124 thay vì 3.124%). Tuyệt đối KHÔNG nhân thêm 100 ở bước cuối cùng của chương trình (KHÔNG dùng multiply(#X, 100) cho các câu hỏi tính phần trăm).
8. CỰC KỲ QUAN TRỌNG: Nếu cần một giá trị cụ thể từ bảng (ví dụ: doanh thu năm 2022), KHÔNG dùng hàm table_xxx. Hãy tự đọc bảng và viết TRỰC TIẾP con số đó vào hàm toán học.
9. CỰC KỲ QUAN TRỌNG: Nếu câu hỏi yêu cầu tính chênh lệch hoặc so sánh đơn thuần mà không có từ 'phần trăm' hoặc '%', chỉ sử dụng duy nhất phép trừ (subtract) — KHÔNG tự động thêm bước chia (divide) để tính tỷ lệ.

Ví dụ đúng:
  subtract(7.758, 7.523), divide(#0, 7.523) (Tính tỷ lệ tăng trưởng phần trăm dưới dạng tỷ lệ thập phân, không nhân 100)
  divide(99782, 2626154) (Tính phần trăm dưới dạng thập phân, không nhân 100)
  multiply(11228, 1.03) (Nhân trực tiếp giá trị tăng trưởng 3% dự phóng)
  table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1) (Tính max/min trên toàn bộ hàng)

LƯU Ý QUAN TRỌNG VỀ ĐỊNH DẠNG CHIẾN LƯỢC:
- Định dạng suy luận (cot_format) có thể chọn từ: "none" (không suy nghĩ trước khi trả lời, direct program), "stepbystep" (suy nghĩ từng bước ngắn gọn), hoặc "chain" (lập luận đầy đủ).
- Trích xuất 1-2 ví dụ few-shot từ Failure Logs (giữ ngắn gọn). Các ví dụ few-shot phải viết theo đúng Cú Pháp Chương Trình ở trên.
- instruction_phrasing là phần hướng dẫn/phong cách/vai trò chung viết bằng tiếng Việt. KHÔNG chứa các chuỗi giữ chỗ như {passage}, {question}, {few_shot_block} vì hệ thống tự động chèn.

YÊU CẦU ĐỘ DÀI VÀ CẤU TRÚC (BẮT BUỘC):
- Viết ngắn gọn và súc tích. Tổng độ dài toàn bộ câu trả lời KHÔNG ĐƯỢC vượt quá 500 từ.
- KHÔNG lặp từ, không giải thích dông dài, không tự tạo ra văn bản rác hoặc ký tự lặp vô nghĩa.
"""


def _is_valid_dsl_program(program: str) -> bool:
    """
    Validate that the program syntax matches the FinQA DSL constraints.
    - Must not contain '=', '+', '*', or '/' (which indicate raw arithmetic equations, not DSL functions).
    - Can contain minus sign '-' if it represents a negative number (e.g., add(-167.4, -53.3)).
    - Must contain at least one valid DSL operator or a step reference (e.g. #0).

    TODO: Implement this validation check.
    """
    if not program or not isinstance(program, str):
        return False

    # Raw arithmetic symbols indicate an equation, not a DSL program.
    if any(sym in program for sym in ("=", "+", "*", "/")):
        return False

    valid_ops = (
        "add", "subtract", "multiply", "divide", "exp", "greater", "abs",
        "table_max", "table_min", "table_sum", "table_average",
    )
    has_op = any(re.search(rf"\b{op}\s*\(", program) for op in valid_ops)
    has_ref = re.search(r"#\d+", program) is not None
    return has_op or has_ref


def _build_propose_message(history: StrategyHistory, parent_strategy_id: Optional[str] = None) -> str:
    lines = ["=== Lịch sử chiến lược ==="]
    for s, r in zip(history.strategies, history.reflections):
        acc = f"{s.metadata.dev_accuracy:.3f}" if s.metadata.dev_accuracy is not None else "chưa đánh giá"
        lines.append(
            f"\nIteration {s.metadata.iteration} | ID: {s.id[:8]} | dev_accuracy={acc} | cot={s.cot_format.value}"
        )
        lines.append(f"  Template: {s.prompt_template[:300]!r}")
        if r is not None:
            lines.append(f"  Loại câu hỏi yếu nhất: {min(r.accuracy_by_type, key=r.accuracy_by_type.get) if r.accuracy_by_type else 'unknown'}")
            lines.append(f"  Giả thuyết: {r.hypothesis[:200]}")

    # Find parent strategy
    parent_strategy = None
    parent_reflection = None
    if parent_strategy_id is not None:
        for s, r in zip(history.strategies, history.reflections):
            if s.id == parent_strategy_id:
                parent_strategy = s
                parent_reflection = r
                break
    if parent_strategy is None:
        parent_strategy = history.latest_strategy()
        parent_reflection = history.latest_reflection()

    if parent_strategy is not None:
        lines.append("\n=== Chiến lược gốc cần tối ưu (Parent Strategy) ===")
        lines.append(f"ID: {parent_strategy.id[:8]}")
        lines.append(f"CoT: {parent_strategy.cot_format.value}")
        lines.append(f"Template:\n{parent_strategy.prompt_template}")
        if parent_reflection is not None:
            lines.append(f"Giả thuyết từ chiến lược gốc: {parent_reflection.hypothesis}")
            lines.append(f"Tóm tắt hiệu suất: {parent_reflection.summary}")

    next_iter = len(history.strategies)
    lines.append(f"\n=== Nhiệm vụ ===")
    lines.append(
        f"Hãy đề xuất một chiến lược mới bằng cách thay đổi/tối ưu trực tiếp từ chiến lược gốc (Parent Strategy: {parent_strategy.id[:8] if parent_strategy else 'None'}). "
        f"Không tối ưu dựa trên các chiến lược khác hoặc chiến lược gần đây nhất nếu nó khác chiến lược gốc này. "
        f"Đề xuất cho iteration {next_iter}."
    )
    return "\n".join(lines)


def generate_few_shot_reasoning(
    passage: str,
    question: str,
    program: str,
    category: str,
    model,  # QwenInference
    max_attempts: int = 3,
) -> str:
    """
    Generate a full CoT-style response for a programmatic few-shot example.
    """
    from src.executor import normalize_program as _norm

    gold_norm = _norm(program)

    system_message = (
        "Bạn là trợ lý AI chuyên phân tích tài chính tiếng Việt. "
        "Nhiệm vụ của bạn là tạo ra một câu trả lời mẫu (few-shot demonstration) "
        "cho bài toán tài chính, theo đúng định dạng đầu ra mà mô hình phải tạo ra.\n\n"
        "Định dạng bắt buộc:\n"
        "1. Một khối <think>...</think> ngắn gọn (tối đa 100 từ), tập trung vào "
        "công thức toán học và các giá trị cần trích xuất. KHÔNG viết dài dòng.\n"
        "2. Ngay sau </think>, một khối JSON với đúng 3 khóa:\n"
        "{\n"
        "  \"Reasoning\": \"Giải thích 2 câu tiếng Việt: câu 1 nêu giá trị trích xuất, câu 2 giải thích phép tính\",\n"
        "  \"Program syntax\": \"<phải khớp CHÍNH XÁC với chương trình đã cho>\",\n"
        "  \"Numerical result\": <kết quả số cuối cùng>\n"
        "}\n\n"
        "QUAN TRỌNG: Trường 'Program syntax' phải chứa ĐÚNG chương trình đã được cung cấp, không thay đổi."
    )

    for attempt in range(max_attempts):
        user_message = (
            f"Ngữ cảnh:\n{passage}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Chương trình đúng: {program}\n\n"
            f"Hãy tạo câu trả lời mẫu hoàn chỉnh theo định dạng <think>...</think>{{JSON}} "
            f"với 'Program syntax' phải là CHÍNH XÁC: {program}"
        )
        prompt = model.format_prompt(
            system_message=system_message,
            user_message=user_message,
            enable_thinking=True,
        )
        try:
            raw_output = model.generate_text(prompt, max_new_tokens=512, temperature=0.0)
            extracted = extract_answer(raw_output)
            if extracted and _norm(extracted) == gold_norm:
                logger.debug(
                    "Few-shot CoT verified on attempt %d (gold=%s extracted=%s)",
                    attempt + 1, program, extracted,
                )
                return raw_output.strip()
            else:
                logger.warning(
                    "Few-shot CoT attempt %d/%d: program mismatch "
                    "(gold_norm=%r, extracted_norm=%r) — retrying",
                    attempt + 1, max_attempts,
                    gold_norm, _norm(extracted) if extracted else None,
                )
        except Exception as e:
            logger.warning("Few-shot CoT generation attempt %d failed: %s", attempt + 1, e)

    logger.warning(
        "All %d attempts failed for program %r — using static fallback reasoning",
        max_attempts, program,
    )
    return f"Bài toán thuộc nhóm {category}. Thực hiện phép tính theo chương trình DSL."


def propose_self(
    history: StrategyHistory,
    model,  # QwenInference
    max_retries: int = 5,
    parent_strategy_id: Optional[str] = None,
    train_dataset: Optional[Dataset] = None,
) -> tuple[Strategy, int]:
    """
    Use the Qwen inference model itself to propose a new strategy.

    TODO: Implement strategy proposal and dynamic few-shot selection.
    Steps:
      1. Build propose message.
      2. Call model to generate a free-form proposal.
      3. Clean thinking tags.
      4. Coerce into a JSON ProposerSchema dictionary.
      5. Identify weakest category from reflection.
      6. Select up to 2 matching training examples and generate CoT reasoning for them.
      7. Validate generated/extracted few-shot programs using _is_valid_dsl_program().
      8. Return a new Strategy object and meta token usage.
    """
    # Resolve the parent strategy and its reflection.
    parent_strategy = None
    parent_reflection = None
    if parent_strategy_id is not None:
        for s, r in zip(history.strategies, history.reflections):
            if s.id == parent_strategy_id:
                parent_strategy = s
                parent_reflection = r
                break
    if parent_strategy is None:
        parent_strategy = history.latest_strategy()
        parent_reflection = history.latest_reflection()
    if parent_strategy is None:
        raise ValueError("Cannot propose a strategy from an empty history.")

    message = _build_propose_message(history, parent_strategy_id=parent_strategy.id)
    next_iter = len(history.strategies)
    total_tokens = 0

    for attempt in range(1, max_retries + 1):
        # Pass 1: free-form proposal.
        prompt = model.format_prompt(
            system_message=_SYSTEM_PROPOSE,
            user_message=message,
            enable_thinking=True,
        )
        raw_proposal = model.generate_text(prompt, max_new_tokens=2048, temperature=0.7)
        total_tokens += model.count_tokens(prompt) + model.count_tokens(raw_proposal)

        # Clean thinking tags.
        proposal = re.sub(r"<think>.*?</think>", "", raw_proposal, flags=re.DOTALL)
        proposal = re.sub(r"<think>.*$", "", proposal, flags=re.DOTALL).strip()

        # Pass 2: coerce the proposal into a strict ProposerSchema JSON.
        coercion_message = (
            "Dưới đây là bản đề xuất chiến lược prompting mới:\n\n"
            f"{proposal}\n\n"
            "Hãy chuyển bản đề xuất trên thành MỘT khối JSON duy nhất theo đúng schema "
            "với các khóa: hypothesis, instruction_phrasing, cot_format "
            "('none' | 'stepbystep' | 'chain'), few_shot_examples "
            "(danh sách {passage, question, answer, reasoning}), reasoning. "
            "Không thêm văn bản nào ngoài khối JSON."
        )
        coercion_prompt = model.format_prompt(
            system_message=_SYSTEM_PROPOSE,
            user_message=coercion_message,
            enable_thinking=False,
        )
        json_text = model.generate_text(
            coercion_prompt,
            max_new_tokens=2048,
            temperature=0.0,
            guided_json=ProposerSchema.model_json_schema(),
        )
        total_tokens += model.count_tokens(coercion_prompt) + model.count_tokens(json_text)

        try:
            match = re.search(r"\{.*\}", json_text, re.DOTALL)
            payload = json.loads(match.group() if match else json_text)
            schema = ProposerSchema.model_validate(payload)
        except Exception as exc:
            logger.warning(
                "Proposal JSON parse failed on attempt %d/%d: %s",
                attempt, max_retries, exc,
            )
            continue

        cot_format = (
            CoTFormat(schema.cot_format)
            if schema.cot_format in _VALID_COT
            else parent_strategy.cot_format
        )

        # Few-shot examples proposed by the meta-agent (validated against the DSL).
        schema_examples: list[FewShotExample] = []
        for ex in schema.few_shot_examples:
            if _is_valid_dsl_program(ex.answer):
                schema_examples.append(
                    FewShotExample(
                        passage=ex.passage,
                        question=ex.question,
                        answer=ex.answer,
                        reasoning=ex.reasoning,
                    )
                )
            else:
                logger.warning("Dropping proposed few-shot with invalid DSL: %r", ex.answer)

        # Dynamic few-shot: target the weakest category from the parent reflection.
        dynamic_examples: list[FewShotExample] = []
        weakest_category = None
        if parent_reflection is not None and parent_reflection.accuracy_by_type:
            weakest_category = min(
                parent_reflection.accuracy_by_type,
                key=parent_reflection.accuracy_by_type.get,
            )
        if weakest_category is not None and train_dataset is not None:
            for row in train_dataset:
                if len(dynamic_examples) >= 2:
                    break
                program = row.get("answer") or ""
                if classify_question_type(program) != weakest_category:
                    continue
                if not _is_valid_dsl_program(program):
                    continue
                passage = row.get("context") or ""
                question = row.get("question") or ""
                reasoning = generate_few_shot_reasoning(
                    passage=passage,
                    question=question,
                    program=program,
                    category=weakest_category,
                    model=model,
                )
                dynamic_examples.append(
                    FewShotExample(
                        passage=passage,
                        question=question,
                        answer=program,
                        reasoning=reasoning,
                    )
                )

        # Merge: parent examples first, then proposed, then dynamic ones.
        merged_examples = (
            list(parent_strategy.few_shot_examples) + schema_examples + dynamic_examples
        )

        new_strategy = Strategy(
            id=str(uuid.uuid4()),
            prompt_template=schema.instruction_phrasing,
            cot_format=cot_format,
            few_shot_examples=merged_examples,
            retrieval_config=RetrievalConfig(
                **parent_strategy.retrieval_config.to_dict()
            ),
            metadata=StrategyMetadata(
                iteration=next_iter,
                parent_id=parent_strategy.id,
            ),
        )
        return new_strategy, total_tokens

    # Fallback: clone the parent with a fresh identity so the loop can continue.
    logger.warning("propose_self falling back to a parent clone after %d failed attempts.", max_retries)
    fallback = Strategy(
        id=str(uuid.uuid4()),
        prompt_template=parent_strategy.prompt_template,
        cot_format=parent_strategy.cot_format,
        few_shot_examples=list(parent_strategy.few_shot_examples),
        retrieval_config=RetrievalConfig(**parent_strategy.retrieval_config.to_dict()),
        metadata=StrategyMetadata(iteration=next_iter, parent_id=parent_strategy.id),
    )
    return fallback, total_tokens
