from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.evaluation.evaluate import evaluate_predictions


def generate_demo_dataset(output_dir: str | Path) -> List[dict]:
    """Create a tiny synthetic dataset that exercises the pipeline.

    The data are intentionally small and deterministic so the prototype can be run
    in a minimal environment without requiring a full image dataset or a trained YOLO model.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset: List[dict] = []
    for idx in range(8):
        image_id = f"demo_{idx:03d}"
        objects = []
        for class_name, box in {
            "worker": [0.15, 0.20, 0.25, 0.35],
            "helmet": [0.22, 0.08, 0.10, 0.14],
            "vest": [0.18, 0.35, 0.30, 0.18],
        }.items():
            objects.append({"class_name": class_name, "bbox": box})

        dataset.append({
            "image_id": image_id,
            "image_path": str(output / f"{image_id}.jpg"),
            "annotations": objects,
        })

    manifest_path = output / "demo_dataset.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)
    return dataset


class SimpleDetector:
    """A tiny placeholder detector used to test the training/inference/evaluation workflow."""

    def fit(self, dataset: Iterable[dict]) -> None:
        self.dataset = list(dataset)

    def predict(self, image_record: dict) -> List[dict]:
        predictions: List[dict] = []
        for annotation in image_record["annotations"]:
            bbox = annotation["bbox"]
            predictions.append({
                "class_name": annotation["class_name"],
                "bbox": [bbox[0] + 0.01, bbox[1] + 0.01, bbox[2], bbox[3]],
                "confidence": 0.90,
            })
        return predictions


def run_demo_pipeline() -> Dict[str, float]:
    dataset = generate_demo_dataset("data/processed/demo")
    detector = SimpleDetector()
    detector.fit(dataset)

    gt_annotations: List[dict] = []
    pred_annotations: List[dict] = []
    for item in dataset[:4]:
        gt_annotations.extend({
            "class_name": ann["class_name"],
            "bbox": ann["bbox"],
        } for ann in item["annotations"])
        pred_annotations.extend({
            "class_name": ann["class_name"],
            "bbox": ann["bbox"],
        } for ann in detector.predict(item))

    metrics = evaluate_predictions(gt_annotations, pred_annotations, iou_threshold=0.5)
    result_path = Path("results/prototype_metrics.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_demo_pipeline()
    print(json.dumps(metrics, indent=2))
