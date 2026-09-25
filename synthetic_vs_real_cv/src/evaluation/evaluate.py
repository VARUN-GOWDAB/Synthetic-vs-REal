from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two [x, y, w, h] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] * box_a[3])
    area_b = max(0.0, box_b[2] * box_b[3])
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def evaluate_predictions(gt_annotations: List[dict], pred_annotations: List[dict], iou_threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, and F1 for a simple detection set."""
    gt_by_class: Dict[str, List[dict]] = {}
    pred_by_class: Dict[str, List[dict]] = {}

    for item in gt_annotations:
        gt_by_class.setdefault(item["class_name"], []).append(item)
    for item in pred_annotations:
        pred_by_class.setdefault(item["class_name"], []).append(item)

    tp = 0
    fp = 0
    fn = 0

    for class_name, gt_items in gt_by_class.items():
        gt_boxes = [item["bbox"] for item in gt_items]
        pred_boxes = [item["bbox"] for item in pred_by_class.get(class_name, [])]
        matched = set()

        for pred_index, pred_box in enumerate(pred_boxes):
            best_iou = -1.0
            best_gt = None
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched:
                    continue
                value = iou(pred_box, gt_box)
                if value > best_iou:
                    best_iou = value
                    best_gt = gt_index
            if best_gt is not None and best_iou >= iou_threshold:
                matched.add(best_gt)
                tp += 1
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    for class_name, pred_items in pred_by_class.items():
        if class_name not in gt_by_class:
            fp += len(pred_items)

    total_predictions = tp + fp
    total_ground_truth = tp + fn
    precision = 0.0 if total_predictions == 0 else tp / total_predictions
    recall = 0.0 if total_ground_truth == 0 else tp / total_ground_truth
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
