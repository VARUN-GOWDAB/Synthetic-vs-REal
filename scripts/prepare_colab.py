"""Freeze a 300-image comparison using existing labels and package it for Colab."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import zipfile

from colab_training import label_counts, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "synthetic_vs_real_cv"
REVIEW = PROJECT / "results/annotation_review_v5"
OUTPUT = PROJECT / "results/colab_preparation_300"
BASELINES = PROJECT / "archive/comparison_300_blender_mixtures_2026-10-07"
SOURCES = ("real_safety_500_v4", "ai_generated_v4", "3d_rendered")


def audited_label_matches(path, expected):
    """Old Windows audit hashes include CRLF; Git checkout uses LF."""
    import hashlib
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(content).hexdigest() for content in
                        (raw, normalized, normalized.replace(b"\n", b"\r\n"))}


def build_manifest(train_images=300, review_policy="existing_labels"):
    if review_policy not in ("existing_labels", "reviewed_only"):
        raise ValueError("Unknown annotation review policy")
    require_review = review_policy == "reviewed_only"
    if train_images < 4 or train_images % 4:
        raise ValueError("Training size must be a positive multiple of four")
    inventory = json.loads((PROJECT / "results/full_visual_review_v4/inventory.json").read_text())
    original_decisions = json.loads((PROJECT / "results/full_visual_review_v4/decisions.json").read_text())
    fresh_real = json.loads((REVIEW / "real_decisions.json").read_text())
    fresh_rendered = json.loads((REVIEW / "rendered_decisions.json").read_text())
    frozen = {(r["source"], r["image"]): r for r in json.loads((PROJECT / "results/comparison_preparation_v4/inventory.json").read_text())}
    scene_groups = {r["image"]: r["scene_group"] for r in json.loads((PROJECT / "results/real_ppe_v2_import/selected_records.json").read_text())}
    reviewed = {(r["source"].replace("_v3", "_v4"), r["image"]): r for r in inventory}
    pools = {s: [] for s in SOURCES}
    holdouts = {"val": [], "test": []}
    exclusions = []
    for source in SOURCES:
        base = PROJECT / "data" / source
        source_manifest = json.loads((base / "annotation_manifest.json").read_text())
        for name in source_manifest["images"]:
            image = base / "images" / name
            label = base / "labels" / (Path(name).stem + ".txt")
            previous = reviewed.get((source, name))
            split = previous["split"] if previous else "synthetic_pool"
            reason = None
            proof = None
            if not image.exists() or not label.exists():
                reason = "Missing source image or matching label"
            elif source == "3d_rendered":
                proof = fresh_rendered.get(name)
                if (proof and proof["status"] != "accepted") or (require_review and not proof):
                    reason = proof["reason"] if proof else "Not accepted in the additional rendered visual review"
            elif source == "real_safety_500_v4":
                proof = fresh_real.get(str(previous["id"]))
                old = original_decisions[str(previous["id"])]
                if proof and proof["status"] != "accepted":
                    reason = proof["reason"]
                elif require_review and old["status"] != "visually_reviewed":
                    reason = "Unresolved original review: " + old.get("notes", "")
                elif (require_review or split in holdouts) and not proof:
                    reason = proof["reason"] if proof else "Not selected for the additional real-image screen"
            else:
                old = original_decisions[str(previous["id"])]
                if require_review and old["status"] != "visually_reviewed":
                    reason = "Unresolved original review: " + old.get("notes", "")
            key = f"{source}/{name}"
            if reason:
                exclusions.append({"id": key, "original_split": split, "reason": reason})
                continue
            # Inherited review decisions apply only to their original image/label
            # contents. Tolerate line-ending conversion, never altered annotations.
            audit = frozen[(source, name)]
            image_hash = sha256(image)
            label_hash = sha256(label)
            inherited = previous and original_decisions[str(previous["id"])]["status"] == "visually_reviewed"
            if image_hash != audit["sha256"] or (not proof and inherited and not audited_label_matches(label, audit["label_sha256"])):
                raise ValueError(f"Source changed since the frozen annotation audit: {key}")
            if proof and (image_hash != proof["image_sha256"] or label_hash != proof["label_sha256"]):
                raise ValueError(f"Source changed since the new visual screen: {key}")
            try:
                objects = label_counts(label.read_text())
            except ValueError as error:
                exclusions.append({"id": key, "original_split": split,
                                   "reason": "Invalid YOLO annotation: " + str(error)})
                continue
            row = {"id": key, "source": source, "image": f"data/{source}/images/{name}",
                   "label": f"data/{source}/labels/{label.name}", "image_sha256": image_hash,
                   "label_sha256": label_hash, "objects": objects, "original_split": split,
                   "review": "additional_visual_screen" if proof else (
                       "inherited_v4_visually_reviewed" if inherited else "existing_labels_not_visually_approved"),
                   "scene_group": f"real:{scene_groups[name]}" if source == SOURCES[0] else None}
            if source == SOURCES[0] and split in holdouts:
                holdouts[split].append(row)
            else:
                pools[source].append(row)
    eligible_counts = {source: len(rows) for source, rows in pools.items()}
    for source, rows in pools.items():
        if len(rows) < train_images:
            raise ValueError(f"Only {len(rows)} eligible images in {source}; requested {train_images}")
        rows.sort(key=lambda r: r["id"])
        random.Random(42).shuffle(rows)
        for unused in rows[train_images:]:
            exclusions.append({"id": unused["id"], "original_split": unused["original_split"],
                               "reason": "Eligible but not sampled for the equal-size comparison"})
        pools[source] = rows[:train_images]
    for rows in holdouts.values():
        rows.sort(key=lambda r: r["id"])
        if not rows:
            raise ValueError("No reviewed real holdout remains")
    real, ai, rendered = SOURCES
    n = train_images
    designs = [("real_only", {real: n}), ("ai_only", {ai: n}), ("rendered_only", {rendered: n}),
               ("mixed_ai50_rendered25_real25", {ai: n // 2, rendered: n // 4, real: n // 4}),
               ("mixed_real50_ai25_rendered25", {real: n // 2, ai: n // 4, rendered: n // 4}),
               ("mixed_ai50_real50", {ai: n // 2, real: n // 2}),
               ("mixed_real75_ai25", {real: 3 * n // 4, ai: n // 4}),
               ("mixed_ai75_real25", {ai: 3 * n // 4, real: n // 4})]
    experiments = []
    for name, counts in designs:
        train = [r for source, count in counts.items() for r in pools[source][:count]]
        random.Random(42).shuffle(train)
        for split, rows in {"train": train, **holdouts}.items():
            if any(sum(r["objects"][str(c)] for r in rows) == 0 for c in range(3)):
                raise ValueError(f"Missing class in {name}/{split}")
        experiments.append({"name": f"{name}_{n}_{review_policy}", "sources": counts,
                            "splits": {"train": [r["id"] for r in train],
                                       **{s: [r["id"] for r in rows] for s, rows in holdouts.items()}}})
    records = sorted([r for rows in pools.values() for r in rows] + [r for rows in holdouts.values() for r in rows], key=lambda r: r["id"])
    manifest = {"version": 1, "name": f"comparison_ai_real_mixtures_{n}_{review_policy}", "ready_for_training": True,
                "annotation_review_policy": review_policy,
                "fully_visually_reviewed": require_review,
                "classes": ["worker", "helmet", "vest"], "seed": 42,
                "train_images_per_experiment": n, "records": records, "experiments": experiments,
                "limitations": [
                    "Existing labels are used without further visual review at the user's request." if not require_review else "Eligibility restricted to previously reviewed images.",
                    "Known rejected images and missing files remain excluded; sampling is selection-biased.",
                    "Training readiness means structural validation, not certification of annotation accuracy.",
                    "Per-image review provenance is retained; unresolved labels may contain errors.",
                    "Very small real validation/test sets and repeated frames within test limit statistical confidence.",
                    "Known visual scene overlaps removed; original scene grouping is heuristic and source video IDs are unavailable.",
                    "Single seed; compare only these eight models on these same holdouts, not bundled v2 metrics."]}
    report = {"status": "comparison_prepared", "original_full_v4_review_complete": False,
              "annotation_review_policy": review_policy,
              "review_provenance_counts": dict(Counter(r["review"] for r in records)),
              "train_images_per_experiment": n, "eligible_train_pool": eligible_counts,
              "holdout_counts": {s: len(rows) for s, rows in holdouts.items()},
              "selected_unique_images": len(records), "excluded_count": len(exclusions),
              "selected_class_counts": {source: dict(sum((Counter(r["objects"]) for r in records if r["source"] == source), Counter())) for source in SOURCES},
              "exclusions": exclusions, "limitations": manifest["limitations"]}
    return manifest, report


def package(manifest, report, archive):
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.writestr("dataset_manifest.json", json.dumps(manifest, indent=2) + "\n")
        bundle.writestr("selection_report.json", json.dumps(report, indent=2) + "\n")
        bundle.write(ROOT / "scripts/colab_training.py", "colab_training.py")
        bundle.write(ROOT / "requirements-colab.txt", "requirements-colab.txt")
        # Include all five completed models for verified reuse.
        for name in ("dataset_manifest.json", "run_specification.json", "initial_weights.json"):
            bundle.write(BASELINES / name, "previous_baselines/" + name)
        for name in ("real_only", "ai_only", "rendered_only",
                     "mixed_ai50_rendered25_real25", "mixed_real50_ai25_rendered25"):
            experiment = name + "_300_existing_labels"
            for relative in ("weights/best.pt", "test_metrics.json"):
                path = experiment + "/" + relative
                bundle.write(BASELINES / path, "previous_baselines/" + path)
        for name in ("real_decisions.json", "rendered_decisions.json", "input_audit.json"):
            bundle.write(REVIEW / name, "review/" + name)
        for row in manifest["records"]:
            for field in ("image", "label"):
                source = PROJECT / row[field]
                if sha256(source) != row[field + "_sha256"]:
                    raise ValueError(f"File changed while packaging: {source}")
                bundle.write(source, row[field])
    temporary.replace(archive)
    return sha256(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=int, default=300)
    parser.add_argument("--review-policy", choices=("existing_labels", "reviewed_only"), default="existing_labels")
    parser.add_argument("--archive", type=Path, default=ROOT / "colab/synthreal_300_ai_real.zip")
    args = parser.parse_args()
    manifest, report = build_manifest(args.train_images, args.review_policy)
    save_json(OUTPUT / "dataset_manifest.json", manifest)
    save_json(OUTPUT / "selection_report.json", report)
    checksum = package(manifest, report, args.archive)
    args.archive.with_suffix(".sha256").write_text(checksum + "  " + args.archive.name + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("exclusions", "limitations")}, indent=2))
    print(f"Archive: {args.archive}\nSHA-256: {checksum}")


if __name__ == "__main__":
    main()
