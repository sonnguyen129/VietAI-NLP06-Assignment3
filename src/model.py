"""
model.py — QwenInference: OpenAI-compatible HTTP client for a remote vLLM server.

Inference runs on a Cerebrium-hosted vLLM OpenAI-compatible server (see
cerebrium.toml). This client keeps the exact same public surface as the
original in-process SGLang wrapper (load / generate_batch / generate_text /
format_prompt / count_tokens), so the rest of the codebase is unchanged.

Configuration comes from the environment (a local .env is honoured):
    CEREBRIUM_BASE_URL  e.g. https://api.cortex.cerebrium.ai/v4/p-xxxx/assignment03/v1
    CEREBRIUM_API_KEY   Cerebrium inference token
    CEREBRIUM_MODEL     served model id (default: QuantTrio/Qwen3.5-4B-AWQ)
    HF_TOKEN            HuggingFace token for downloading the tokenizer

Answer extraction handles extraction of mathematical programs:
    e.g., "subtract(108.50, 100), divide(#0, 100)"
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Holds the raw text output and token counts for a single example."""

    raw_output: str
    predicted_answer: Optional[str]  # "A", "B", "C", "D", or None
    input_tokens: int
    output_tokens: int


class QwenInference:
    """
    OpenAI-compatible HTTP inference client for Qwen3.5-Instruct models
    served by a remote vLLM server (deployed on Cerebrium).

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID. Used to download the tokenizer locally and as
        the served model id unless CEREBRIUM_MODEL overrides it.
    max_new_tokens:
        Maximum tokens generated per example.
    temperature:
        Sampling temperature. 0.0 = greedy decoding.
    use_4bit / gpu_memory_utilization / max_model_len:
        Kept for CLI compatibility. GPU-related values are applied on the
        server (cerebrium.toml); max_model_len is still used for local
        prompt truncation.
    base_url / api_key:
        OpenAI-compatible endpoint and token. Default to the
        CEREBRIUM_BASE_URL / CEREBRIUM_API_KEY environment variables.
    request_timeout:
        Per-request timeout in seconds (long CoT generations can be slow,
        especially on a cold-started replica).
    max_concurrency:
        Number of parallel HTTP requests in generate_batch(). vLLM
        continuous-batches server-side, so this just keeps the pipe full.
    """

    def __init__(
        self,
        model_name_or_path: str = "QuantTrio/Qwen3.5-4B-AWQ",

        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_4bit: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: float = 600.0,
        max_concurrency: int = 8,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.base_url = base_url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_concurrency = max_concurrency

        self._client = None
        self._served_model: Optional[str] = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Connect to the remote vLLM server and load the local tokenizer."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        from openai import OpenAI
        from transformers import AutoTokenizer

        base_url = self.base_url or os.environ.get("CEREBRIUM_BASE_URL")
        api_key = self.api_key or os.environ.get("CEREBRIUM_API_KEY") or "EMPTY"
        if not base_url:
            raise RuntimeError(
                "No inference endpoint configured. Set CEREBRIUM_BASE_URL "
                "(and CEREBRIUM_API_KEY) in the environment or .env file, "
                "or pass base_url= to QwenInference."
            )
        self._served_model = os.environ.get("CEREBRIUM_MODEL") or self.model_name_or_path

        logger.info(
            "Connecting to vLLM server: base_url=%s model=%s",
            base_url,
            self._served_model,
        )
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.request_timeout,
            max_retries=2,
        )

        # Load tokenizer locally for apply_chat_template / count_tokens.
        # Tokenizer-only download — no GPU or torch weights required.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN") or None,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Warm-up call: fail fast if the endpoint is unreachable and wake a
        # scaled-to-zero replica before the first real batch.
        try:
            models = self._client.models.list()
            served = [m.id for m in models.data]
            logger.info("Server reachable. Served models: %s", served)
            if self._served_model not in served and served:
                logger.warning(
                    "Configured model %r not in served list %s — using %r.",
                    self._served_model, served, served[0],
                )
                self._served_model = served[0]
        except Exception as exc:
            logger.warning("Warm-up models.list() failed (continuing): %s", exc)

        logger.info("Inference client ready.")

    def _complete(self, prompt: str, sampling_params: dict) -> tuple[str, int, int]:
        """
        Single /v1/completions request. Returns (text, input_tokens, output_tokens).

        vLLM-specific sampling knobs (top_k, min_p, guided_json, ...) travel in
        extra_body — isolated here so a vLLM API rename only touches one place.
        """
        params = dict(sampling_params)
        extra_body = {}
        for key in ("top_k", "min_p", "repetition_penalty", "guided_json"):
            if key in params:
                value = params.pop(key)
                if value is not None:
                    extra_body[key] = value

        response = self._client.completions.create(
            model=self._served_model,
            prompt=prompt,
            max_tokens=params.pop("max_new_tokens"),
            temperature=params.pop("temperature"),
            top_p=params.pop("top_p"),
            presence_penalty=params.pop("presence_penalty"),
            stop=params.pop("stop"),
            extra_body=extra_body or None,
        )
        text = response.choices[0].text or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else self.count_tokens(prompt)
        output_tokens = usage.completion_tokens if usage else self.count_tokens(text)
        return text, input_tokens, output_tokens

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate_batch(self, prompts: list[str], cot_format: bool = False) -> list[GenerationResult]:
        """
        Run generation for a list of already-formatted prompt strings.

        Requests are fanned out over max_concurrency threads; the vLLM server
        continuous-batches them on the GPU for maximum throughput.
        """
        if self._client is None:
            raise RuntimeError("Call load() before generate_batch().")

        if self.temperature > 0.0:
            if cot_format:
                # Thinking mode for precise coding tasks (e.g. CoT Program Generation)
                temp = 0.6 if self.temperature == 1.0 or self.temperature == 0.7 else self.temperature
                t_p = 0.95
                t_k = 20
                m_p = 0.0
                p_p = 0.0
            else:
                # Instruct (or non-thinking) mode for reasoning tasks
                temp = self.temperature
                t_p = 0.95
                t_k = 20
                m_p = 0.0
                p_p = 1.5
        else:
            temp = 0.0
            t_p = 1.0
            t_k = -1
            m_p = 0.0
            p_p = 0.0

        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temp,
            "top_p": t_p,
            "top_k": t_k,
            "min_p": m_p,
            "presence_penalty": p_p,
            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        }

        # Truncate prompts that exceed the context budget.
        max_input = self.max_model_len - self.max_new_tokens
        if max_input <= 0:
            raise ValueError(
                f"max_model_len ({self.max_model_len}) must be strictly greater than "
                f"max_new_tokens ({self.max_new_tokens}) to leave room for the prompt."
            )
        truncated = []
        for p in prompts:
            ids = self._tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > max_input:
                ids = ids[:max_input]
                p = self._tokenizer.decode(ids, skip_special_tokens=True)
            truncated.append(p)

        def run_one(prompt: str) -> tuple[str, int, int]:
            try:
                return self._complete(prompt, sampling_params)
            except Exception as exc:
                logger.error("Completion request failed: %s", exc)
                return "", self.count_tokens(prompt), 0

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            outputs = list(pool.map(run_one, truncated))

        results: list[GenerationResult] = []
        for raw_text, input_tokens, output_tokens in outputs:
            raw_text = raw_text.strip()
            predicted = extract_answer(raw_text)
            results.append(
                GenerationResult(
                    raw_output=raw_text,
                    predicted_answer=predicted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def format_prompt(
        self,
        system_message: str,
        user_message: str,
        enable_thinking: bool = False,
    ) -> str:
        """
        Build a ChatML-formatted string for Qwen-Instruct using the
        tokenizer's apply_chat_template().

        For Qwen3 models, enable_thinking=False (default) injects the /no_think
        control token which disables the verbose reasoning block and produces
        direct answers — critical for eval throughput and token budget.
        Set enable_thinking=True only for meta-agent tasks (reflection/proposing)
        where reasoning quality matters more than brevity.
        """
        if self._tokenizer is None:
            raise RuntimeError("Call load() before format_prompt().")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers (Qwen2.5) don't support enable_thinking — fall back
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        guided_json: Optional[dict | str] = None,
    ) -> str:
        """
        Single unconstrained text generation — used by self_proposer / self_reflector.

        Unlike generate_batch() which is optimised for high-throughput MC inference,
        this is for low-volume, long-form generation (strategy proposals, reflections).
        Returns the raw generated text string.
        """
        if self._client is None:
            raise RuntimeError("Call load() before generate_text().")

        # For guided JSON, enforce deterministic decoding
        if guided_json is not None:
            temp = 0.0
            t_p = 1.0
            t_k = -1
            m_p = 0.0
            p_p = 0.0
        else:
            temp = temperature
            t_p = top_p
            t_k = top_k
            m_p = min_p
            p_p = presence_penalty

        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temp,
            "top_p": t_p,
            "top_k": t_k,
            "min_p": m_p,
            "presence_penalty": p_p,
            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        }

        if guided_json is not None:
            # vLLM accepts the JSON schema as a dict or string via guided_json.
            sampling_params["guided_json"] = (
                guided_json if isinstance(guided_json, dict) else json.loads(guided_json)
            )

        max_input = self.max_model_len - max_new_tokens
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input:
            ids = ids[:max_input]
            prompt = self._tokenizer.decode(ids, skip_special_tokens=True)

        text, _, _ = self._complete(prompt, sampling_params)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before count_tokens().")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def is_loaded(self) -> bool:
        return self._client is not None


# ------------------------------------------------------------------
# Answer extraction (module-level for testability)
# ------------------------------------------------------------------


def clean_unnecessary_parentheses(program: str) -> str:
    if not program:
        return program
    chars = list(program)
    stack = []
    pairs = []
    for idx, char in enumerate(chars):
        if char == '(':
            stack.append(idx)
        elif char == ')':
            if stack:
                start_idx = stack.pop()
                pairs.append((start_idx, idx))
    to_delete = set()
    for start, end in pairs:
        prev_idx = start - 1
        while prev_idx >= 0 and chars[prev_idx].isspace():
            prev_idx -= 1
        is_func_call = False
        if prev_idx >= 0:
            c = chars[prev_idx]
            if c.isalnum() or c == '_':
                is_func_call = True
        if not is_func_call:
            to_delete.add(start)
            to_delete.add(end)
    cleaned = "".join(chars[i] for i in range(len(chars)) if i not in to_delete)
    cleaned = re.sub(r',\s*,', ', ', cleaned)
    cleaned = cleaned.strip().strip(',')
    return cleaned

def fix_bare_hashes(program: str) -> str:
    if not program:
        return program
    chars = list(program)
    step_count = 0
    i = 0
    while i < len(chars):
        if chars[i] == ')':
            step_count += 1
        elif chars[i] == '#':
            if i + 1 >= len(chars) or not chars[i + 1].isdigit():
                prev_step_idx = max(0, step_count - 1)
                replacement = str(prev_step_idx)
                chars.insert(i + 1, replacement)
                i += len(replacement)
        i += 1
    return "".join(chars)

def clean_and_fix_program(program: str) -> str:
    if not program:
        return program
    program = clean_unnecessary_parentheses(program)
    program = fix_bare_hashes(program)
    return program

def extract_answer(text: str) -> Optional[str]:
    """
    Extract a mathematical program string from a model's free-form output.

    Handles multi-line programs, balanced nested parentheses, prefixes,
    and Qwen3-style thinking blocks (<think>...</think> or "Thinking Process:").
    """
    if not text:
        return None

    # First check if there is a </think> block.
    # If so, the JSON block must be after the </think> tag.
    content_after_think = text
    if "</think>" in text:
        parts = text.split("</think>")
        content_after_think = parts[-1].strip()

    # Try to extract from JSON format first
    json_match = re.search(r"\{.*\}", content_after_think, re.DOTALL)
    if json_match:
        try:
            import json
            data = json.loads(json_match.group(), strict=False)
            program_val = None
            for key in ["Program syntax", "program syntax", "Program", "program", "program_syntax"]:
                if key in data:
                    program_val = data[key]
                    break
            if not program_val:
                for key, val in data.items():
                    if "program" in key.lower() or "syntax" in key.lower():
                        program_val = val
                        break
            if program_val and isinstance(program_val, str):
                program_val = program_val.strip()
                program_val = re.sub(r"^`+", "", program_val)
                program_val = re.sub(r"`+$", "", program_val).strip()
                if program_val:
                    return clean_and_fix_program(program_val)
        except Exception:
            pass

    # Fallback: operate on content_after_think for the rest of extraction
    text = content_after_think

    # ---- 0. Strip Qwen3 thinking blocks ----
    # Remove <think>...</think> blocks (Qwen3 native thinking format)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    # ---- High-priority strict PROGRAM: extraction ----
    program_match = re.search(r"PROGRAM:\s*(.*)", text, re.IGNORECASE)
    if program_match:
        # Extract the line containing the program
        prog_text = program_match.group(1).strip().split("\n")[0].strip()
        # Clean markdown backticks if any
        prog_text = re.sub(r"^`+", "", prog_text)
        prog_text = re.sub(r"`+$", "", prog_text).strip()
        if prog_text:
            return clean_and_fix_program(prog_text)

    # If there's an explicit label like "Chương trình: X" or "**Answer:**\n X",
    # extract only what comes after it. This handles Thinking Process blocks that
    # end with a clearly labeled answer.
    explicit_label = re.search(
        r"(?:ch\u01b0\u01a1ng\s*tr\u00ecnh|program|\*\*answer\*\*|\*\*\u0111\u00e1p\s*\u00e1n\*\*)\s*[:\-]?\s*\n*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit_label:
        text = explicit_label.group(1).strip()
    else:
        # Greedy strip: if "Thinking Process:" is present, drop everything up to
        # a double newline that *immediately* precedes a line starting with a
        # known operation (i.e. the final answer paragraph).
        if re.search(r"Thinking Process:", text, re.IGNORECASE):
            # Try to find the LAST paragraph that contains only program tokens
            # (no bullet points, no markdown headers)
            paragraphs = re.split(r"\n{2,}", text)
            clean_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                # Keep paragraphs that look like programs (function calls) and
                # NOT like thinking step headers (numbered lists, markdown headers)
                if para and not re.match(r"^(\d+\.|#+|\*\*|Thinking Process)", para):
                    clean_paragraphs.append(para)
            text = "\n\n".join(clean_paragraphs) if clean_paragraphs else text

    text = text.strip()

    if not text:
        return None

    # Operations list
    ops = {"add", "subtract", "multiply", "divide", "table_average", "table_max", "table_min", "table_sum", "abs"}

    # 1. Strip markdown code blocks
    text = re.sub(r"```(?:[a-zA-Z_0-9]*)\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # 2. Clean up "Chương trình:", "Program:", "Đáp án:", "Answer:" prefixes
    text = re.sub(
        r"^(?:chương\s*trình|program|đáp\s*án|answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 3. Parse line by line to extract valid program components
    lines = text.split("\n")
    valid_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that look like backtick-formatted function lists
        # (system-prompt leakage: "`add`, `subtract`, `multiply`...")
        backtick_ops = re.findall(r"`([a-zA-Z_]+)`", line)
        if len(backtick_ops) >= 2 and all(
            op in ops for op in backtick_ops
        ):
            continue

        # Look for operations or references
        has_op = any(op in line.lower() for op in ops)
        has_ref = any(f"#{i}" in line for i in range(10))

        if has_op or has_ref:
            # Find the starting index of the expression (first operation word or reference/parenthesis)
            first_idx = len(line)
            for op in ops:
                idx = line.lower().find(op)
                if idx != -1 and idx < first_idx:
                    first_idx = idx

            if first_idx == len(line):
                # Search for any operation prefix or opening parenthesis
                op_match = re.search(r"[a-zA-Z_0-9]+(?=\()", line)
                if op_match:
                    first_idx = op_match.start()
                else:
                    first_idx = 0

            # Adjust first_idx backwards to include any leading parentheses/spaces/commas
            while first_idx > 0 and line[first_idx - 1] in {'(', ' ', ','}:
                if line[first_idx - 1].isalnum() or line[first_idx - 1] == '_':
                    break
                first_idx -= 1

            part = line[first_idx:].strip()
            part = clean_and_fix_program(part)
            if part:
                # We scan character by character to extract balanced chunks
                stack = 0
                chunk_start = 0
                has_paren = False
                for i, char in enumerate(part):
                    if char == '(':
                        stack += 1
                        has_paren = True
                    elif char == ')':
                        stack -= 1
                        has_paren = True
                        if stack == 0:
                            # We found a complete balanced chunk!
                            chunk = part[chunk_start:i+1].strip()
                            if chunk:
                                valid_parts.append(chunk)
                            # Now search for the next operation start after this chunk
                            next_idx = i + 1
                            while next_idx < len(part) and part[next_idx] in {',', ' ', '\n', '\t'}:
                                next_idx += 1
                            chunk_start = next_idx

                # Fallback: if we didn't find any balanced paren but operation keywords are present
                if not has_paren:
                    valid_parts.append(part)

    if valid_parts:
        return clean_and_fix_program(", ".join(valid_parts))

    # Fallback: Check if the text itself matches a program-like layout
    if "(" in text and ")" in text:
        return clean_and_fix_program(text.strip())

    return None

