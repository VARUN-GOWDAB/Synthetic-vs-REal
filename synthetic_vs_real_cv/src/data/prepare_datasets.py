from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REAL_IMAGE_ROOT = Path("data/real/images")
REAL_LABEL_ROOT = Path("data/real/labels")
SYNTHETIC_IMAGE_ROOT = Path("data/synthetic/images")
SYNTHETIC_LABEL_ROOT = Path("data/synthetic/labels")
PROCESSED_ROOT = Path("data/processed")
EXPERIMENTS = [
    {"name": "100_real", "real_ratio": 1.0, "synthetic_ratio": 0.0},
    {"name": "75_real_25_synthetic", "real_ratio": 0.75, "synthetic_ratio": 0.25},
    {"name": "50_real_50_synthetic", "real_ratio": 0.5, "synthetic_ratio": 0.5},
    {"name": "25_real_75_synthetic", "real_ratio": 0.25, "synthetic_ratio": 0.75},
    {"name": "100_synthetic", "real_ratio": 0.0, "synthetic_ratio": 1.0},
]


def list_image_files(image_dir: Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[Path]:
    if not image_dir.exists():
        return []
    files = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return files


def split_files(files: List[Path], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    if not files:
        return [], [], []

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    if total == 1:
        return [shuffled[0]], [], []
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))
    train_count = total - val_count - test_count

    if train_count <= 0:
        train_count = max(1, total - val_count)
        test_count = max(0, total - train_count - val_count)

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return train, val, test


def copy_label_if_exists(image_path: Path, source_label_dir: Path, destination_label_dir: Path) -> None:
    label_path = source_label_dir / f"{image_path.stem}.txt"
    if label_path.exists():
        destination_label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, destination_label_dir / label_path.name)


def prepare_dataset_split(image_dir: Path, label_dir: Path, output_root: Path, dataset_name: str, seed: int = 42) -> Dict[str, int]:
    image_files = list_image_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    train_files, val_files, test_files = split_files(image_files, val_ratio=0.15, test_ratio=0.15, seed=seed)

    split_map = {"train": train_files, "val": val_files, "test": test_files}
    for split_name, split_items in split_map.items():
        split_image_dir = output_root / dataset_name / split_name / "images"
        split_label_dir = output_root / dataset_name / split_name / "labels"
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_items:
            target_image = split_image_dir / image_path.name
            shutil.copy2(image_path, target_image)
            copy_label_if_exists(image_path, label_dir, split_label_dir)

    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def build_experiment_train_sets(real_train_root: Path, synthetic_train_root: Path, output_root: Path) -> List[Dict[str, object]]:
    real_images = list_image_files(real_train_root / "images")
    synthetic_images = list_image_files(synthetic_train_root / "images")

    manifests: List[Dict[str, object]] = []
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        real_ratio = float(experiment["real_ratio"])
        synthetic_ratio = float(experiment["synthetic_ratio"])

        real_count = int(round(len(real_images) * real_ratio)) if real_ratio > 0 else 0
        synthetic_count = int(round(len(synthetic_images) * synthetic_ratio)) if synthetic_ratio > 0 else 0

        target_train_dir = output_root / name / "train"
        target_images_dir = target_train_dir / "images"
        target_labels_dir = target_train_dir / "labels"
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        real_selected = real_images[:real_count]
        synthetic_selected = synthetic_images[:synthetic_count]

        for image_path in real_selected:
            target_path = target_images_dir / image_path.name
            shutil.copy2(image_path, target_path)
            label_source = real_train_root / "labels" / f"{image_path.stem}.txt"
            if label_source.exists():
                shutil.copy2(label_source, target_labels_dir / label_source.name)

        for image_path in synthetic_selected:
            target_path = target_images_dir / image_path.name
            shutil.copy2(image_path, target_path)
            label_source = synthetic_train_root / "labels" / f"{image_path.stem}.txt"
            if label_source.exists():
                shutil.copy2(label_source, target_labels_dir / label_source.name)

        manifests.append({
            "name": name,
            "real_ratio": real_ratio,
            "synthetic_ratio": synthetic_ratio,
            "real_count": real_count,
            "synthetic_count": synthetic_count,
            "total_train_images": real_count + synthetic_count,
            "train_dir": str(target_train_dir),
        })

    summary_path = output_root / "experiment_manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)

    return manifests


def prepare_full_dataset_flow() -> Dict[str, object]:
    real_split = prepare_dataset_split(REAL_IMAGE_ROOT, REAL_LABEL_ROOT, PROCESSED_ROOT, "real")
    synthetic_split = prepare_dataset_split(SYNTHETIC_IMAGE_ROOT, SYNTHETIC_LABEL_ROOT, PROCESSED_ROOT, "synthetic")
    experiment_summary = build_experiment_train_sets(PROCESSED_ROOT / "real" / "train", PROCESSED_ROOT / "synthetic" / "train", PROCESSED_ROOT)

    return {
        "real_split": real_split,
        "synthetic_split": synthetic_split,
        "experiments": experiment_summary,
    }


if __name__ == "__main__":
    result = prepare_full_dataset_flow()
    print(json.dumps(result, indent=2))
