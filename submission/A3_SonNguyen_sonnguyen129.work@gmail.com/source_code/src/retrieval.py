"""
retrieval.py — Phase-3 kNN few-shot retrieval over train.json.

TF-IDF char n-grams (diacritic-robust for Vietnamese, no torch). For each
dev/test question we retrieve similar train questions and use their GOLD flat
programs as per-question few-shot examples (teaching the exact executable
format for free).

Strategy B ("retr") = the s4fix base template with its static few-shots
swapped for these retrieved ones — a single-variable change vs strategy A.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.resolve()

N_SHOTS = 3
MAX_PASSAGE_CHARS = 500
MAX_PROMPT_TOKENS = 6000  # guard handled by caller-side truncation of shots


def _op_signature(program: str) -> str:
    return ",".join(re.findall(r"([a-zA-Z_]+)\s*\(", program or ""))


@dataclass
class TrainIndex:
    rows: list[dict]
    vectorizer: object = None
    matrix: object = None
    _cache: dict = field(default_factory=dict)

    @classmethod
    def build_default(cls) -> "TrainIndex":
        from src.data import load_dataset

        rows = [r for r in load_dataset()["train"] if r.get("answer")]
        index = cls(rows=rows)
        index.build()
        return index

    def build(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=200_000, lowercase=True
        )
        self.matrix = self.vectorizer.fit_transform([r["question"] for r in self.rows])
        logger.info("TrainIndex built over %d train questions.", len(self.rows))

    def topk(self, question: str, k: int = 8) -> list[dict]:
        import numpy as np

        query = self.vectorizer.transform([question])
        scores = (self.matrix @ query.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [self.rows[i] for i in top]


def _shot_passage(row: dict) -> str:
    """Compact passage: window of the train context around the gold operands."""
    context = row.get("context") or ""
    program = row.get("answer") or ""
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", program)
    for number in numbers:
        plain = number.replace(",", "")
        pos = context.find(plain)
        if pos >= 0:
            start = max(0, pos - MAX_PASSAGE_CHARS // 2)
            return context[start:start + MAX_PASSAGE_CHARS]
    return context[:MAX_PASSAGE_CHARS]


def make_few_shots(question: str, index: TrainIndex, n_shots: int = N_SHOTS):
    """Top-k retrieve, then greedily pick n_shots maximizing op-signature diversity."""
    from src.strategy import FewShotExample

    pool = index.topk(question, k=max(8, n_shots * 3))
    picked, seen_sigs = [], set()
    for row in pool:
        sig = _op_signature(row["answer"])
        if sig in seen_sigs and len(pool) - pool.index(row) > (n_shots - len(picked)):
            continue
        picked.append(row)
        seen_sigs.add(sig)
        if len(picked) >= n_shots:
            break
    for row in pool:  # fill up if diversity filter was too strict
        if len(picked) >= n_shots:
            break
        if row not in picked:
            picked.append(row)

    shots = []
    for row in picked:
        shots.append(FewShotExample(
            passage=_shot_passage(row),
            question=row["question"],
            answer=row["answer"],
            reasoning=f"Trích xuất các giá trị liên quan từ ngữ cảnh rồi áp dụng: {row['answer']}.",
        ))
    return shots


def strategy_for_row(base_strategy, row: dict, index: TrainIndex):
    """Clone the base strategy with retrieved few-shots. strategy_id stays 'retr'
    at the cache layer; the clone id only matters for prompt construction."""
    from src.strategy import Strategy

    if row["id"] in index._cache:
        shots = index._cache[row["id"]]
    else:
        shots = make_few_shots(row["question"], index)
        index._cache[row["id"]] = shots

    return Strategy(
        id=base_strategy.id + "-retr",
        prompt_template=base_strategy.prompt_template,
        cot_format=base_strategy.cot_format,
        few_shot_examples=shots,
        retrieval_config=base_strategy.retrieval_config,
        metadata=base_strategy.metadata,
    )
