"""
data.py — Load dataset splits for EvoAgent.

Loads from the bundled data/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default: bundled data next to this file or parent directory
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
if not _DEFAULT_DATA_DIR.exists() and (Path(__file__).parent.parent / "data").exists():
    _DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def _format_table(table: list[list[str]]) -> str:
    """Format a 2D list table into a simple text-based table representation."""
    if not table:
        return ""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def _load_json_split(path: Path) -> list[dict]:
    """
    Parse a program-generation format JSON file into a flat list of examples.

    Each example has: id, context (pre_text + table + post_text), question, answer (program), exe_ans.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        pre_text = item.get("pre_text") or []
        table = item.get("table") or []
        post_text = item.get("post_text") or []
        
        pre_str = " ".join(pre_text).strip()
        post_str = " ".join(post_text).strip()
        table_str = _format_table(table).strip()
        
        context_parts = []
        if pre_str:
            context_parts.append(pre_str)
        if table_str:
            context_parts.append("Bảng:\n" + table_str)
        if post_str:
            context_parts.append(post_str)
            
        context = "\n\n".join(context_parts).strip()
        
        qa = item.get("qa") or {}
        question = qa.get("question", "").strip()
        program = qa.get("program", "").strip()
        exe_ans = str(qa.get("exe_ans", "")).strip()

        examples.append({
            "id": item.get("id", str(len(examples))),
            "context": context,
            "question": question,
            "options": [],
            "answer": program,  # Gold program stored in 'answer' for compatibility
            "exe_ans": exe_ans,
            "question_type": "",
            "table": table,
        })
    return examples


def load_dataset(data_dir: Optional[Path] = None) -> dict:
    """
    Load data splits from local JSON files.

    Returns a HuggingFace DatasetDict with train / validation / test splits.
    """
    from datasets import Dataset, DatasetDict

    data_dir = Path(data_dir or _DEFAULT_DATA_DIR)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    splits = {}
    file_map = {
        "train": data_dir / "train.json",
        "validation": data_dir / "dev.json",
        "test": data_dir / "test.json",
    }

    for split, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        examples = _load_json_split(path)
        splits[split] = Dataset.from_list(examples)
        logger.info("Loaded %d examples for %s split.", len(examples), split)

    return DatasetDict(splits)


def load_data_splits(
    train_size: int = 100,
    dev_size: Optional[int] = None,
    seed: int = 42,
    data_dir: Optional[Path] = None,
):
    """
    Load data and return (train_subset, dev_split).

    Parameters
    ----------
    train_size: number of train examples for cheap mid-loop eval.
    dev_size: number of dev examples (None = full dev split).
    seed: random seed for shuffling.
    data_dir: path to directory containing train.json / dev.json / test.json.
    """
    ds = load_dataset(data_dir=data_dir)

    train_split = ds["train"].shuffle(seed=seed)
    train_subset = train_split.select(range(min(train_size, len(train_split))))

    dev_split = ds["validation"].shuffle(seed=seed)
    if dev_size is not None:
        dev_split = dev_split.select(range(min(dev_size, len(dev_split))))

    logger.info(
        "Splits ready: train_subset=%d, dev=%d.",
        len(train_subset), len(dev_split),
    )
    return train_subset, dev_split

