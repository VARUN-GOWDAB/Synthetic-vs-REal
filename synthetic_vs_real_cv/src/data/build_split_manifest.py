from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def build_split_manifest(
    dataset_name: str,
    train_count: int,
    val_count: int,
    test_count: int,
    classes: List[str],
    source: str,
    output_path: str | Path,
) -> Dict[str, object]:
    manifest = {
        "dataset_name": dataset_name,
        "source": source,
        "classes": classes,
        "splits": {
            "train": {
                "count": train_count,
                "purpose": "training data",
            },
            "validation": {
                "count": val_count,
                "purpose": "model selection and early stopping",
            },
            "test": {
                "count": test_count,
                "purpose": "final unseen real-world evaluation",
            },
        },
        "split_policy": {
            "leakage_control": [
                "group by video, camera, or near-duplicate scene",
                "keep test set fully separate from training",
                "never reuse same scene or location across train and test",
            ],
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


if __name__ == "__main__":
    real_manifest = build_split_manifest(
        dataset_name="real_industrial_safety",
        train_count=700,
        val_count=150,
        test_count=150,
        classes=["worker", "helmet"],
        source="Safety Helmet Wearing Dataset (candidate)",
        output_path="data/processed/real_split_manifest.json",
    )
    synthetic_manifest = build_split_manifest(
        dataset_name="synthetic_industrial_safety",
        train_count=800,
        val_count=100,
        test_count=100,
        classes=["worker", "helmet"],
        source="Blender Python domain-randomized synthetic dataset",
        output_path="data/processed/synthetic_split_manifest.json",
    )

    print(json.dumps({"real": real_manifest["splits"], "synthetic": synthetic_manifest["splits"]}, indent=2))
