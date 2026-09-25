from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json


def build_training_config() -> Dict[str, Any]:
    return {
        "model": {
            "name": "yolov8n.pt",
            "architecture": "YOLOv8n",
            "image_size": 640,
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.01,
            "optimizer": "AdamW",
            "patience": 20,
            "seed": 42,
        },
        "data": {
            "train": "data/processed/train",
            "val": "data/processed/val",
            "test": "data/processed/test",
            "classes": ["worker", "helmet"],
        },
        "training": {
            "device": "0",
            "project": "results/yolo_runs",
            "name": "synthetic_vs_real_experiment",
            "exist_ok": True,
        },
        "evaluation": {
            "metrics": ["precision", "recall", "mAP50", "mAP50_95"],
            "iou_threshold": 0.5,
        },
    }


def save_training_config(output_path: str | Path) -> Dict[str, Any]:
    config = build_training_config()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config


if __name__ == "__main__":
    config = save_training_config("configs/yolo_training_config.json")
    print(json.dumps(config, indent=2))
