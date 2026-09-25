from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_experiment_runs() -> List[Dict[str, Any]]:
    runs = [
        {
            "name": "100_real",
            "real_count": 100,
            "synthetic_count": 0,
            "real_ratio": 1.0,
            "synthetic_ratio": 0.0,
            "train_path": "data/processed/real/train",
            "val_path": "data/processed/real/val",
            "test_path": "data/processed/real/test",
        },
        {
            "name": "75_real_25_synthetic",
            "real_count": 75,
            "synthetic_count": 25,
            "real_ratio": 0.75,
            "synthetic_ratio": 0.25,
            "train_path": "data/processed/mixed_75_25/train",
            "val_path": "data/processed/real/val",
            "test_path": "data/processed/real/test",
        },
        {
            "name": "50_real_50_synthetic",
            "real_count": 50,
            "synthetic_count": 50,
            "real_ratio": 0.5,
            "synthetic_ratio": 0.5,
            "train_path": "data/processed/mixed_50_50/train",
            "val_path": "data/processed/real/val",
            "test_path": "data/processed/real/test",
        },
        {
            "name": "25_real_75_synthetic",
            "real_count": 25,
            "synthetic_count": 75,
            "real_ratio": 0.25,
            "synthetic_ratio": 0.75,
            "train_path": "data/processed/mixed_25_75/train",
            "val_path": "data/processed/real/val",
            "test_path": "data/processed/real/test",
        },
        {
            "name": "100_synthetic",
            "real_count": 0,
            "synthetic_count": 100,
            "real_ratio": 0.0,
            "synthetic_ratio": 1.0,
            "train_path": "data/processed/synthetic/train",
            "val_path": "data/processed/synthetic/val",
            "test_path": "data/processed/real/test",
        },
    ]
    return runs


def save_experiment_runs(output_path: str | Path) -> List[Dict[str, Any]]:
    runs = build_experiment_runs()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=2)
    return runs


if __name__ == "__main__":
    runs = save_experiment_runs("data/processed/experiment_runs.json")
    print(json.dumps(runs, indent=2))
