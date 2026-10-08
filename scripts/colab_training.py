"""Portable, GPU-only training for a frozen SynthReal Colab bundle.

Dataset preparation uses the standard library and Pillow. Torch and Ultralytics
are imported only when training is explicitly requested in Colab.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import time
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def safe_path(root, relative):
    relative = str(relative)
    posix = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    result = (Path(root) / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes dataset: {relative}")
    return result


def extract_bundle(archive, destination):
    """Extract into a new directory; reject traversal and symlink entries."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Already extracted: {destination}. Reuse it or choose a new directory.")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            safe_path(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Symlink in bundle: {member.filename}")
        destination.mkdir(parents=True)
        bundle.extractall(destination)
    return destination


def label_counts(text):
    counts = {"0": 0, "1": 0, "2": 0}
    seen = set()
    for line in text.splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Expected five YOLO values: {line}")
        c, x, y, w, h = map(float, values)
        if not all(math.isfinite(v) for v in (c, x, y, w, h)):
            raise ValueError("Non-finite label value")
        if c not in (0, 1, 2) or min(w, h) <= 0:
            raise ValueError(f"Invalid class or dimensions: {line}")
        if min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1.00001:
            raise ValueError(f"Box outside image: {line}")
        key = tuple(map(float, values))
        if key in seen:
            raise ValueError(f"Exact duplicate annotation: {line}")
        seen.add(key)
        counts[str(int(c))] += 1
    return counts


def verify_and_prepare(root):
    """Verify frozen files and memberships, then create machine-local YOLO paths."""
    from PIL import Image

    root = Path(root).resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if manifest.get("ready_for_training") is not True:
        raise ValueError("This bundle is not ready for training.")
    if manifest.get("classes") != ["worker", "helmet", "vest"]:
        raise ValueError("Unexpected class mapping")
    records = manifest["records"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate record ID")
    lookup = {r["id"]: r for r in records}
    image_hashes = set()
    for row in records:
        image = safe_path(root, row["image"])
        label = safe_path(root, row["label"])
        expected_label = image.parent.parent / "labels" / (image.stem + ".txt")
        if image.parent.name != "images" or label != expected_label:
            raise ValueError(f"Image/label layout mismatch: {row['id']}")
        if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError(f"File changed since dataset freeze: {row['id']}")
        if row["image_sha256"] in image_hashes:
            raise ValueError(f"Duplicate image contents: {row['id']}")
        image_hashes.add(row["image_sha256"])
        with Image.open(image) as picture:
            picture.load()
        if label_counts(label.read_text()) != row["objects"]:
            raise ValueError(f"Label counts changed: {row['id']}")
    shared = None
    used = set()
    prepared = []
    for experiment in manifest["experiments"]:
        if not experiment["name"].replace("_", "").isalnum():
            raise ValueError("Invalid experiment name")
        splits = experiment["splits"]
        for split in ("train", "val", "test"):
            ids = splits[split]
            if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(lookup):
                raise ValueError(f"Invalid {split} membership: {experiment['name']}")
            for key in ids:
                row = lookup[key]
                if split != "train" and (row["source"] != "real_safety_500_v4" or row["original_split"] != split):
                    raise ValueError("Holdout contains synthetic data or reassigned images")
                if split == "train" and row["original_split"] in ("val", "test"):
                    raise ValueError("Holdout image used for training")
            used.update(ids)
        sets = [set(splits[s]) for s in ("train", "val", "test")]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Train/validation/test overlap")
        # Check inherited scene groups as well as exact image hashes. Manual scene
        # exclusions in the review ledger address known failures of those groups.
        groups = [{lookup[k].get("scene_group") for k in ids if lookup[k].get("scene_group") is not None} for ids in sets]
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Known source scene group crosses splits")
        holdouts = (splits["val"], splits["test"])
        if shared is not None and holdouts != shared:
            raise ValueError("Experiments do not share identical real holdouts")
        shared = holdouts
        if len(splits["train"]) != manifest["train_images_per_experiment"]:
            raise ValueError("Unequal training-set sizes")
        source_counts = {}
        for key in splits["train"]:
            source = lookup[key]["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        if source_counts != experiment["sources"]:
            raise ValueError("Experiment source ratios differ from the manifest")
        directory = root / "prepared" / experiment["name"]
        directory.mkdir(parents=True, exist_ok=True)
        for split, ids in splits.items():
            (directory / f"{split}.txt").write_text("\n".join(str(safe_path(root, lookup[k]["image"])) for k in ids) + "\n")
        # JSON strings are also valid YAML strings, including paths with spaces.
        yaml = f"path: {json.dumps(str(directory))}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [worker, helmet, vest]\n"
        (directory / "data.yaml").write_text(yaml)
        prepared.append({**experiment, "data": str(directory / "data.yaml")})
    if used != set(lookup):
        raise ValueError("Manifest contains unused or unaccounted records")
    return {"root": str(root), "manifest_sha256": sha256(root / "dataset_manifest.json"),
            "manifest": manifest, "experiments": prepared}


def freeze_run(output, specification):
    output = Path(output)
    path = output / "run_specification.json"
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError("Dataset or settings changed. Use a new run folder; do not mix comparisons.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty run folder without a specification. Use a new folder.")
    else:
        save_json(path, specification)


def reuse_completed_baselines(prepared, output, previous, specification, initialization_sha):
    """Import unchanged completed models after verifying their provenance and bytes."""
    previous, output = Path(previous), Path(output)
    old_manifest_path = previous / "dataset_manifest.json"
    old = json.loads(old_manifest_path.read_text())
    old_spec = json.loads((previous / "run_specification.json").read_text())
    if old_spec["dataset_sha256"] != sha256(old_manifest_path):
        raise ValueError("Previous manifest checksum mismatch")
    if any(old_spec[key] != specification[key] for key in ("settings", "ultralytics")):
        raise ValueError("Baseline settings differ; disable baseline reuse for a fresh comparison")
    if json.loads((previous / "initial_weights.json").read_text())["sha256"] != initialization_sha:
        raise ValueError("Baseline initialization differs")
    current = prepared["manifest"]
    if any(old[key] != current[key] for key in ("classes", "seed")):
        raise ValueError("Baseline classes or seed differ")
    old_records = {r["id"]: r for r in old["records"]}
    current_records = {r["id"]: r for r in current["records"]}
    old_experiments = {e["name"]: e for e in old["experiments"]}
    pending = []
    for experiment in current["experiments"]:
        name = experiment["name"]
        if name not in old_experiments:
            continue
        if old_experiments.get(name) != experiment:
            raise ValueError(f"Baseline membership changed: {name}")
        ids = {key for split in experiment["splits"].values() for key in split}
        if any(old_records.get(key) != current_records[key] for key in ids):
            raise ValueError(f"Baseline data changed: {name}")
        source = previous / name
        result = json.loads((source / "test_metrics.json").read_text())
        if (result["experiment"] != name or result["sources"] != experiment["sources"] or
                any(result[f"{split}_images"] != len(experiment["splits"][split])
                    for split in ("train", "val", "test"))):
            raise ValueError(f"Baseline metrics metadata differs: {name}")
        if sha256(source / "weights/best.pt") != result["checkpoint_sha256"]:
            raise ValueError(f"Baseline checkpoint checksum mismatch: {name}")
        destination = output / name
        for relative in ("weights/best.pt", "test_metrics.json"):
            if (destination / relative).exists() and sha256(destination / relative) != sha256(source / relative):
                raise ValueError(f"Existing baseline output differs: {name}/{relative}")
        pending.append((name, source, result))
    # Validate all baselines before copying any of them.
    for name, source, result in pending:
        for relative in ("weights/best.pt", "test_metrics.json"):
            destination = output / name / relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".importing")
                shutil.copy2(source / relative, temporary)
                temporary.replace(destination)
    save_json(output / "reused_baselines.json", {
        "source_manifest_sha256": sha256(old_manifest_path),
        "source_specification": old_spec,
        "models": {name: result["checkpoint_sha256"] for name, _, result in pending},
        "note": "Previously evaluated models copied unchanged; no new training or evaluation performed."})
    return [name for name, _, _ in pending]


def run_training(prepared, output, epochs=50, batch=8, reuse_baselines=True):
    """Run or resume the comparison. Never silently fall back to CPU."""
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch, int) or batch < 1:
        raise ValueError("epochs and batch must be positive integers")
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU attached. In Colab select Runtime > Change runtime type > GPU, then rerun setup.")
    if ultralytics.__version__ != "8.4.173":
        raise RuntimeError("Install the notebook's pinned ultralytics==8.4.173 dependency first.")
    # Recheck data immediately before consuming GPU time.
    prepared = verify_and_prepare(prepared["root"])
    output = Path(output).resolve()
    settings = {"model": "yolov8n.pt", "epochs": epochs, "batch": batch, "imgsz": 640,
                "seed": 42, "optimizer": "AdamW", "lr0": 0.001, "patience": 0,
                "device": 0, "workers": 2, "deterministic": True, "amp": True,
                "cache": False, "save": True, "save_period": 10, "plots": True}
    specification = {"dataset_sha256": prepared["manifest_sha256"], "settings": settings,
                     "ultralytics": ultralytics.__version__}
    freeze_run(output, specification)
    shutil.copy2(Path(prepared["root"]) / "dataset_manifest.json", output / "dataset_manifest.json")
    save_json(output / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
              "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0), "time": time.ctime()})
    weights = output / "initial_weights"
    weights.mkdir(exist_ok=True)
    initial = weights / "yolov8n.pt"
    if not initial.exists():
        previous_directory = Path.cwd()
        try:
            os.chdir(weights)
            YOLO("yolov8n.pt")  # Download the official initialization once, on Colab.
        finally:
            os.chdir(previous_directory)
    weight_record = output / "initial_weights.json"
    checksum = {"sha256": sha256(initial)}
    if weight_record.exists() and json.loads(weight_record.read_text()) != checksum:
        raise ValueError("Initialization checkpoint changed")
    save_json(weight_record, checksum)
    previous = Path(prepared["root"]) / "previous_baselines"
    if reuse_baselines and previous.exists():
        reused = reuse_completed_baselines(prepared, output, previous, specification, checksum["sha256"])
        print("Reusing verified completed models:", ", ".join(reused), flush=True)
    scores = []
    try:
        for experiment in prepared["experiments"]:
            name = experiment["name"]
            run = output / name
            result_path = run / "test_metrics.json"
            best = run / "weights/best.pt"
            last = run / "weights/last.pt"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if not best.exists() or sha256(best) != result["checkpoint_sha256"]:
                    raise ValueError(f"Completed checkpoint missing/changed: {name}")
                scores.append(result)
                print(f"Already complete: {name}", flush=True)
                continue
            finished = False
            model = None
            if last.exists():
                # Only load checkpoints produced by this run in the user's Drive.
                checkpoint = torch.load(last, map_location="cpu", weights_only=False)
                finished = checkpoint["epoch"] == -1 or checkpoint["epoch"] + 1 >= epochs
                if not finished:
                    model = YOLO(str(last))
                del checkpoint
            elif run.exists() and any(run.iterdir()):
                raise RuntimeError(f"{name} has output but no last.pt. Restore a saved checkpoint or use a new run folder.")
            if not finished:
                save_json(output / "status.json", {"state": "training", "experiment": name, "time": time.ctime()})
                if model is not None:
                    model.train(resume=True, device=0)
                else:
                    model = YOLO(str(initial))
                    model.train(data=experiment["data"], project=str(output), name=name,
                                exist_ok=False, **{k: v for k, v in settings.items() if k != "model"})
            if not best.exists():
                raise FileNotFoundError(f"Training did not produce {best}")
            save_json(output / "status.json", {"state": "evaluating", "experiment": name, "time": time.ctime()})
            model = YOLO(str(best))
            metrics = model.val(data=experiment["data"], split="test", device=0,
                                imgsz=640, batch=batch, workers=2, project=str(output),
                                name=name + "_test", exist_ok=True, plots=True)
            result = {"experiment": name, "train_images": len(experiment["splits"]["train"]),
                      "val_images": len(experiment["splits"]["val"]), "test_images": len(experiment["splits"]["test"]),
                      "sources": experiment["sources"], "checkpoint_sha256": sha256(best),
                      "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
                      "per_class_ap50_95": {model.names[int(c)]: float(ap) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap)}}
            save_json(result_path, result)
            scores.append(result)
            write_scores(output, scores)
            del model
            torch.cuda.empty_cache()
        write_scores(output, scores)
        save_json(output / "status.json", {"state": "complete", "experiments": len(scores), "time": time.ctime()})
    except BaseException as error:
        save_json(output / "status.json", {"state": "interrupted", "error": str(error), "time": time.ctime()})
        raise
    return scores


def write_scores(output, scores):
    rows = [{"experiment": r["experiment"], "train_images": r["train_images"],
             "val_images": r["val_images"], "test_images": r["test_images"], **r["metrics"]} for r in scores]
    if not rows:
        return
    path = Path(output) / "comparison_metrics.csv"
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)
