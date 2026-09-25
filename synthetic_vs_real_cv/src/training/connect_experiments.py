from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REAL_TOTAL = 700
DEFAULT_SYNTHETIC_TOTAL = 800


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_matrix() -> List[Dict[str, float]]:
    return [
        {"name": "100_real", "real_ratio": 1.0, "synthetic_ratio": 0.0},
        {"name": "75_real_25_synthetic", "real_ratio": 0.75, "synthetic_ratio": 0.25},
        {"name": "50_real_50_synthetic", "real_ratio": 0.5, "synthetic_ratio": 0.5},
        {"name": "25_real_75_synthetic", "real_ratio": 0.25, "synthetic_ratio": 0.75},
        {"name": "100_synthetic", "real_ratio": 0.0, "synthetic_ratio": 1.0},
    ]


def connect_experiments(
    real_total: int = DEFAULT_REAL_TOTAL,
    synthetic_total: int = DEFAULT_SYNTHETIC_TOTAL,
    real_manifest_path: str | Path = "data/processed/real_split_manifest.json",
    synthetic_manifest_path: str | Path = "data/processed/synthetic_split_manifest.json",
    output_dir: str | Path = "data/processed/experiment_connections",
) -> Dict[str, Any]:
    real_manifest = load_json(real_manifest_path) if Path(real_manifest_path).exists() else {"splits": {"train": {"count": real_total}}}
    synthetic_manifest = load_json(synthetic_manifest_path) if Path(synthetic_manifest_path).exists() else {"splits": {"train": {"count": synthetic_total}}}

    real_train_total = int(real_manifest.get("splits", {}).get("train", {}).get("count", real_total))
    synthetic_train_total = int(synthetic_manifest.get("splits", {}).get("train", {}).get("count", synthetic_total))

    matrix = build_matrix()
    summaries: List[Dict[str, Any]] = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for entry in matrix:
        name = entry["name"]
        real_ratio = float(entry["real_ratio"])
        synthetic_ratio = float(entry["synthetic_ratio"])

        real_count = int(round(real_train_total * real_ratio)) if real_ratio > 0 else 0
        synthetic_count = int(round(synthetic_train_total * synthetic_ratio)) if synthetic_ratio > 0 else 0

        run_manifest = {
            "name": name,
            "real_dataset": {
                "source": "real_split_manifest.json",
                "train_count": real_count,
                "total_available": real_train_total,
            },
            "synthetic_dataset": {
                "source": "synthetic_split_manifest.json",
                "train_count": synthetic_count,
                "total_available": synthetic_train_total,
            },
            "experiment": {
                "real_ratio": real_ratio,
                "synthetic_ratio": synthetic_ratio,
                "total_train_images": real_count + synthetic_count,
            },
            "paths": {
                "train_dir": f"data/processed/{name}/train",
                "val_dir": f"data/processed/{name}/val",
                "test_dir": "data/processed/real/test",
            },
            "status": "ready_for_training",
        }

        run_dir = output / name
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run_connection_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(run_manifest, handle, indent=2)

        summaries.append(run_manifest)

    summary_path = output / "experiment_connection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return {"experiments": summaries, "summary_path": str(summary_path)}


if __name__ == "__main__":
    result = connect_experiments()
    print(json.dumps(result, indent=2))
