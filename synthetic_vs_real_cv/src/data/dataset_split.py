from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json


def load_manifest(manifest_path: str | Path) -> List[dict]:
    """Load a JSON manifest containing image metadata."""
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("images", [])


def split_by_group(entries: Iterable[dict], val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split entries by a grouping field while preserving deterministic ordering."""
    items = list(entries)
    if not items:
        return [], [], []

    total = len(items)
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))
    train_count = total - val_count - test_count

    if train_count <= 0:
        train_count = 1
        val_count = max(1, min(val_count, total - train_count))
        test_count = total - train_count - val_count

    train = items[:train_count]
    val = items[train_count : train_count + val_count]
    test = items[train_count + val_count :]
    return train, val, test


def write_manifest(entries: List[dict], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
