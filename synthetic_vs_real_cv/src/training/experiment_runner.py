from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_experiment_matrix() -> List[Dict[str, float]]:
    return [
        {"name": "100_real", "real_ratio": 1.0, "synthetic_ratio": 0.0},
        {"name": "75_real_25_synthetic", "real_ratio": 0.75, "synthetic_ratio": 0.25},
        {"name": "50_real_50_synthetic", "real_ratio": 0.5, "synthetic_ratio": 0.5},
        {"name": "25_real_75_synthetic", "real_ratio": 0.25, "synthetic_ratio": 0.75},
        {"name": "100_synthetic", "real_ratio": 0.0, "synthetic_ratio": 1.0},
    ]


def save_experiment_matrix(output_path: str | Path) -> List[Dict[str, float]]:
    matrix = build_experiment_matrix()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(matrix, handle, indent=2)
    return matrix


if __name__ == "__main__":
    matrix = save_experiment_matrix("data/processed/experiment_matrix.json")
    print(json.dumps(matrix, indent=2))
