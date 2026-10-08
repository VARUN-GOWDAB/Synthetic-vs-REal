"""Prepare the frozen SynthReal experiment manifest for Ultralytics YOLO.

This script shows the data-preparation step used before model training. It
copies the manifest's exact image/label memberships into YOLO's expected
images/{train,val,test} and labels/{train,val,test} folder layout and writes
one data YAML per experiment.

It does not resize images or augment them. Ultralytics handles model-side
image loading and transforms during training.

Example, when the original Colab bundle has been extracted:
    python training/preprocess_dataset.py --dataset-root /path/to/extracted_bundle

The dataset bundle is not required for the dashboard. Importing this module
does not prepare data; the work only starts when the script is run directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "deployment" / "evaluation" / "dataset_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "prepared"
EXPECTED_CLASSES = ["worker", "helmet", "vest"]
EXPECTED_SPLITS = {"train": 300, "val": 7, "test": 22}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest in manageable chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_manifest_path(root: Path, relative_path: str) -> Path:
    """Resolve one manifest path and prevent it escaping the dataset folder."""
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path leaves the dataset folder: {relative_path}")
    return path


def validate_yolo_label(path: Path) -> None:
    """Check the expected class ID and normalized YOLO box format."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO label values.")
        try:
            class_id = int(values[0])
            center_x, center_y, width, height = map(float, values[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: label values must be numeric.") from error

        if class_id not in range(len(EXPECTED_CLASSES)):
            raise ValueError(f"{path}:{line_number}: unknown class ID {class_id}.")
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
            raise ValueError(f"{path}:{line_number}: box center must be normalized to 0..1.")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: box width and height must be in (0, 1].")


def write_data_yaml(path: Path, experiment_dir: Path) -> None:
    """Write a small YAML config without requiring a YAML package here."""
    root = json.dumps(experiment_dir.resolve().as_posix())
    names = ", ".join(json.dumps(name) for name in EXPECTED_CLASSES)
    path.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )


def prepare(manifest_path: Path, dataset_root: Path, output_root: Path) -> None:
    """Validate the frozen manifest and copy each experiment's exact splits."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classes") != EXPECTED_CLASSES:
        raise ValueError("Manifest classes do not match worker, helmet, vest.")

    records: dict[str, dict[str, Any]] = {}
    for record in manifest.get("records", []):
        identifier = record.get("id")
        if not identifier or identifier in records:
            raise ValueError(f"Missing or duplicate image ID in manifest: {identifier}")
        records[identifier] = record

    experiments = manifest.get("experiments", [])
    if not experiments:
        raise ValueError("The manifest does not contain experiment split memberships.")

    reference_holdouts: dict[str, list[str]] | None = None
    output_root.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        name = experiment["name"]
        splits = experiment["splits"]
        if set(splits) != set(EXPECTED_SPLITS):
            raise ValueError(f"{name}: expected train, val, and test split memberships.")

        for split, expected_count in EXPECTED_SPLITS.items():
            if len(splits[split]) != expected_count:
                raise ValueError(
                    f"{name}: expected {expected_count} {split} images, "
                    f"found {len(splits[split])}."
                )

        if reference_holdouts is None:
            reference_holdouts = {key: splits[key] for key in ("val", "test")}
        elif any(splits[key] != reference_holdouts[key] for key in ("val", "test")):
            raise ValueError(f"{name}: validation/test memberships differ from the shared holdouts.")

        all_ids = [identifier for split_ids in splits.values() for identifier in split_ids]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError(f"{name}: an image occurs in more than one split.")

        experiment_dir = output_root / name
        for split, identifiers in splits.items():
            for identifier in identifiers:
                record = records.get(identifier)
                if record is None:
                    raise ValueError(f"{name}: image is missing from records: {identifier}")

                source_image = safe_manifest_path(dataset_root, record["image"])
                source_label = safe_manifest_path(dataset_root, record["label"])
                if not source_image.is_file() or not source_label.is_file():
                    raise FileNotFoundError(
                        f"Missing image/label pair for {identifier}. "
                        "Restore the matching dataset bundle before preparing data."
                    )

                if record.get("image_sha256") and sha256(source_image) != record["image_sha256"]:
                    raise ValueError(f"Image checksum mismatch: {identifier}")
                if record.get("label_sha256") and sha256(source_label) != record["label_sha256"]:
                    raise ValueError(f"Label checksum mismatch: {identifier}")
                validate_yolo_label(source_label)

                source_name = Path(identifier).parent.name
                image_name = f"{source_name}__{Path(record['image']).name}"
                label_name = f"{source_name}__{Path(record['label']).name}"
                image_target = experiment_dir / "images" / split / image_name
                label_target = experiment_dir / "labels" / split / label_name
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, image_target)
                shutil.copy2(source_label, label_target)

        write_data_yaml(experiment_dir / "data.yaml", experiment_dir)
        print(
            f"Prepared {name}: "
            f"{len(splits['train'])} train, {len(splits['val'])} validation, "
            f"{len(splits['test'])} test images"
        )

    print(f"YOLO-ready datasets and YAML files are in: {output_root.resolve()}")
    print("These structural checks do not prove that annotation boxes are correct.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder of the extracted dataset bundle (contains the manifest's data/ paths).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.manifest, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
